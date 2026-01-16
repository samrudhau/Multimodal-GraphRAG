import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict

import ollama
from pydantic import BaseModel, Field
from neo4j import GraphDatabase, Driver

CHECKPOINT_FILE = "checkpoint_chunks.json"

# --------------------------------------------------
# Utilities
# --------------------------------------------------

def read_processed_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = name.lower()
    n = re.sub(r"\(.*?\)", "", n)
    n = re.sub(r"[^a-z0-9\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"\b(ltd|inc|corp|company|co|llc|the)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n

def is_alias(a: str, b: str) -> bool:
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    ta = set(na.split())
    tb = set(nb.split())
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / max(len(ta), len(tb))
    return overlap >= 0.6

def chunk_text_by_words(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# --------------------------------------------------
# Ollama wrapper
# --------------------------------------------------

class OllamaClientWrapper:
    def __init__(self, model: str = "qwen2.5:7b"):
        host = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        self.client = ollama.Client(host=host)
        self.model = model
        print(f"[ollama] host={host} model={model}")

    def extract_graph_json(self, text: str, max_retries: int = 3) -> dict:
        system = (
            "You extract entities and relationships from text into JSON.\n"
            "Always return JSON with keys: nodes, relationships.\n"
            "nodes = list of {type,name,title}.\n"
            "relationships = list of {source,target,type}.\n"
            "If unsure, guess the type.\n"
            "DO NOT return explanations.\n"
            "Example output:\n"
            "{\n"
            "  \"nodes\": [\n"
            "    {\"type\": \"Organization\", \"name\": \"NVIDIA\", \"title\": \"\"},\n"
            "    {\"type\": \"Speaker\", \"name\": \"Jensen Huang\", \"title\": \"CEO\"}\n"
            "  ],\n"
            "  \"relationships\": [\n"
            "    {\"source\": \"Jensen Huang\", \"target\": \"NVIDIA\", \"type\": \"LEADS\"}\n"
            "  ]\n"
            "}\n"
        )

        user = f"Extract entities and relationships from this text:\n{text}\nReturn JSON only."

        for attempt in range(max_retries):
            try:
                resp = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    format="json"
                )

                js = extract_json_block(resp["message"]["content"])
                data = json.loads(js)

                if "nodes" not in data or "relationships" not in data:
                    raise ValueError("JSON missing keys")

                return data

            except Exception as e:
                print(f"[ollama] extraction failed attempt {attempt+1}: {e}")
                time.sleep(1)

        # final fallback
        return {"nodes": [], "relationships": []}
    
    def embed(self, text: str):
        return self.client.embeddings(
            model="nomic-embed-text",
            prompt=text
        )["embedding"]

def extract_json_block(t: str) -> str:
    a = t.find("{")
    b = t.rfind("}")
    if a == -1 or b == -1:
        return "{}"
    return t[a:b+1]

# --------------------------------------------------
# Graph cleaning/merging
# --------------------------------------------------

def clean_raw_graph(raw):
    nodes = []
    rels = []

    for n in raw.get("nodes", []):
        if not isinstance(n, dict):
            continue
        name = n.get("name")
        if not isinstance(name, str):
            continue

        typ = n.get("type", "KeyConcept")
        nodes.append({
            "type": typ,
            "name": name.strip(),
            "title": n.get("title", "")
        })

    for r in raw.get("relationships", []):
        if not isinstance(r, dict):
            continue
        s = r.get("source")
        t = r.get("target")
        typ = r.get("type")
        if isinstance(s, str) and isinstance(t, str) and isinstance(typ, str):
            rels.append({
                "source": s.strip(),
                "target": t.strip(),
                "type": typ.upper().replace(" ", "_")
            })

    return {"nodes": nodes, "relationships": rels}

def build_alias_map(nodes):
    names = list({n["name"] for n in nodes})
    amap = {}

    for name in sorted(names, key=lambda s: -len(s)):
        if name not in amap:
            amap[name] = name
            for other in names:
                if other not in amap and is_alias(name, other):
                    amap[other] = name

    return amap

def remap_graph(graph, amap):
    out_nodes = []
    seen = set()

    for n in graph["nodes"]:
        canon = amap.get(n["name"], n["name"])
        key = (n["type"], canon)
        if key not in seen:
            out_nodes.append({"type": n["type"], "name": canon, "title": n["title"]})
            seen.add(key)

    out_rels = []
    seenr = set()

    for r in graph["relationships"]:
        s = amap.get(r["source"], r["source"])
        t = amap.get(r["target"], r["target"])
        key = (s, t, r["type"])
        if key not in seenr:
            out_rels.append({"source": s, "target": t, "type": r["type"]})
            seenr.add(key)

    return {"nodes": out_nodes, "relationships": out_rels}

def merge_graphs(graphs):
    nodes = []
    rels = []
    for g in graphs:
        nodes.extend(g["nodes"])
        rels.extend(g["relationships"])
    amap = build_alias_map(nodes)
    return remap_graph({"nodes": nodes, "relationships": rels}, amap)

def sanitize_label(label: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", label.replace(" ", "_"))
    if not label[0].isalpha():
        label = "L_" + label
    return label

# --------------------------------------------------
# Neo4j DB
# --------------------------------------------------

class Neo4jDB:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def import_kg(self, graph):
        with self.driver.session() as s:
            for n in graph["nodes"]:
                label = sanitize_label(n["type"])
                s.run(
                    f"MERGE (x:{label} {{name:$name}}) SET x.title=$title",
                    name=n["name"],
                    title=n.get("title", "")
                )

            for r in graph["relationships"]:
                rtype = sanitize_label(r["type"])
                s.run(
                    f"""
                    MATCH (a {{name:$s}}), (b {{name:$t}})
                    MERGE (a)-[:{rtype}]->(b)
                    """,
                    s=r["source"], t=r["target"]
                )

    def import_chunks(self, chunks, embeddings):
        with self.driver.session() as s:
            for cid, (text, emb) in enumerate(zip(chunks, embeddings)):
                s.run(
                    """
                    MERGE (c:Chunk {id:$id})
                    SET c.text = $text, c.embedding = $embed
                    """,
                    id=f"chunk_{cid}",
                    text=text,
                    embed=emb
                )

# --------------------------------------------------
# Main Pipeline
# --------------------------------------------------

def main():
    # Load processed file
    base = Path(__file__).resolve().parent.parent
    data = read_processed_json(base/"data/processed/NVDA-Q1-2025_processed.json")

    audio = " ".join(s["text"] for s in data["whisper_transcription"]["segments"])
    pdf = data.get("pdf_text", "")
    merged_text = audio + "\n" + pdf

    chunks = chunk_text_by_words(merged_text, 120)
    print(f"[pipeline] chunks = {len(chunks)}")

    client = OllamaClientWrapper()
    per_chunk_graphs = []

    # Entity extraction per chunk
    for idx, c in enumerate(chunks):
        print(f"[chunk {idx+1}/{len(chunks)}]")
        raw = client.extract_graph_json(c)
        cleaned = clean_raw_graph(raw) 
        per_chunk_graphs.append(cleaned) 
        print(f"[chunk] nodes {len(cleaned['nodes'])} rels {len(cleaned['relationships'])}")

    # Merge KG
    merged_kg = merge_graphs(per_chunk_graphs)
    print(f"[KG] nodes={len(merged_kg['nodes'])} rels={len(merged_kg['relationships'])}")

    # Chunk embeddings
    embeddings = []
    for c in chunks:
        embeddings.append(client.embed(c))

    # Import into Neo4j
    db = Neo4jDB()
    db.import_kg(merged_kg)
    db.import_chunks(chunks, embeddings)
    db.close()

    print("[pipeline] complete")

if __name__ == "__main__":
    main()
