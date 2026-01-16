import os
import json
import ollama
from neo4j import GraphDatabase

class Embedder:
    def __init__(self):
        host = os.getenv("OLLAMA_HOST","http://ollama:11434")
        self.client = ollama.Client(host=host)
        self.model = os.getenv("OLLAMA_EMBED_MODEL","nomic-embed-text")

    def encode(self, text):
        return self.client.embeddings(model=self.model, prompt=text)["embedding"]

class KG:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def search_chunks(self, vec, k):
        with self.driver.session() as s:
            q = s.run(
                """
                CALL db.index.vector.queryNodes(
                    'KG_CHUNK_VECTOR', $k, $vec
                ) YIELD node, score
                RETURN node.id AS id, node.text AS text, score
                """,
                k=k,
                vec=vec
            )
            return [dict(r) for r in q]

    def expand(self, cid):
        with self.driver.session() as s:
            q = s.run(
                """
                MATCH (c:Chunk {id:$id})<-[:MENTIONED_IN]-(e:Entity)
                OPTIONAL MATCH (e)-[r]-(y:Entity)
                RETURN e.name AS src, type(r) AS rel, y.name AS tgt
                """,
                id=cid
            )
            triples = []
            for r in q:
                if r["rel"] and r["tgt"]:
                    triples.append({"source":r["src"],"type":r["rel"],"target":r["tgt"]})
            return triples

class LLM:
    def __init__(self):
        host = os.getenv("OLLAMA_HOST","http://ollama:11434")
        self.client = ollama.Client(host=host)
        self.model = os.getenv("OLLAMA_MODEL","qwen2.5:7b")

    def answer(self, query, ctx):
        sys = (
            "Answer strictly from provided context. "
            "Cite chunks and entities in a final 'Sources' section."
        )

        usr = "Query:\n" + query + "\n\nContext:\n" + json.dumps(ctx, indent=1)

        r = self.client.chat(
            model=self.model,
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":usr}],
            format=""
        )
        return r["message"]["content"]

def run_query(query):
    emb = Embedder()
    graph = KG()
    llm = LLM()

    qv = emb.encode(query)
    hits = graph.search_chunks(qv, k=5)

    all_edges = []
    for h in hits:
        edges = graph.expand(h["id"])
        all_edges.extend(edges)

    graph.close()

    ctx = {"chunks": hits, "edges": all_edges}
    return llm.answer(query, ctx)
