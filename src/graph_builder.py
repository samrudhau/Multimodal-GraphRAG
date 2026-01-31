import os
import glob
from pathlib import Path
import json
from typing import List, Dict, Any
import time
import random
from azure.core.exceptions import HttpResponseError
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from neo4j import GraphDatabase
from pydantic import BaseModel, Field

# --- 1. Comprehensive Schema ---
class Entity(BaseModel):
    name: str = Field(description="Normalized name (e.g., 'NVIDIA', 'Revenue', 'Blackwell')")
    type: str = Field(description="PERSON, PRODUCT, COMPANY, METRIC, DATE, SEGMENT, etc.")
    value: str = Field(default="", description="If METRIC, the specific value like '$26B' or '18%'")

class Relationship(BaseModel):
    source: str
    target: str
    relation: str = Field(description="Examples: ANNOUNCED, GREW_BY, REPORTED_BY, IS_PRODUCT_OF, etc.")

class GraphResponse(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)

class GraphBuilder:
    def __init__(self):
        print("🔗 Initializing GitHub Models & Neo4j Driver...", flush=True)
        self.client = ChatCompletionsClient(
            endpoint="https://models.inference.ai.azure.com",
            credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN"))
        )
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://graphrag_db:7687"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def _extract_graph_data(self, text_chunk: str) -> GraphResponse:
        system_prompt = """
        You are a Financial Knowledge Architect. Your goal is to map out a dense web of information.
        
        POSSIBILITIES TO CAPTURE:
        1. Entity Aliases: Always normalize to the official name (e.g. 'Colette' -> 'Colette Kress').
        2. Metric Nodes: Treat specific numbers as 'METRIC' nodes so we can hop between 'Revenue' and its 'Value'.
        3. Strategic Links: Connect products to their specific market segments.
        4. Temporal Context: Link events to their quarters or dates.
        5. Hierarchical Relations: Capture 'IS_PRODUCT_OF', 'IS_DIVISION_OF' to build company structures.
        6. Causal/Comparative Relations: Capture 'GREW_BY', 'DECREASED_DUE_TO' to reflect financial dynamics.

        CRITICAL RULES:
        1. Always return a JSON object with BOTH "entities" and "relationships" keys.
        2. If no relationships are found, return "relationships": [].
        3. If no entities are found, return "entities": [].
        4. Use consistent names (e.g. 'NVIDIA' instead of 'Nvidia Corp').
        RETURN FORMAT:
        {
            "entities": [
                {
                    "name": "NVIDIA",
                    "type": "COMPANY",
                    "value": ""
                },
                {
                    "name": "Revenue",
                    "type": "METRIC",
                    "value": "$26B"
                }
            ],
            "relationships": [
                {
                    "source": "NVIDIA",
                    "target": "Revenue",
                    "relation": "REPORTED"
                }
            ]
        """
        for attempt in range(retries):
            try:
                response = self.client.complete(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text_chunk}
                    ],
                    model="gpt-4o",
                    temperature=0.1
                )
                
                content = response.choices[0].message.content.strip()
                clean_content = content.replace("```json", "").replace("```", "").strip()
                return GraphResponse.model_validate_json(clean_content)

            except HttpResponseError as e:
                if e.status_code == 429:
                    # Rate limit hit! Wait and try again.
                    wait_time = (2 ** attempt) + random.random()
                    print(f"⚠️ Rate limited. Waiting {wait_time:.2f}s before retry {attempt+1}/{retries}...", flush=True)
                    time.sleep(wait_time)
                else:
                    print(f"❌ API Error: {e}", flush=True)
                    break # Critical error, don't retry
            except Exception as e:
                print(f"⚠️ Chunk failed validation: {e}. Skipping...", flush=True)
                break

        # If all retries fail, return empty so the script continues
        return GraphResponse(entities=[], relationships=[])

    def build_graph_from_files(self, base_dir: str = "data/processed"):
        """Recursively finds and processes all JSON chunks."""
        processed_path = Path(base_dir)
        # Recursive glob to find every JSON file in NVIDIA/Q1-2025/ etc.
        files = list(processed_path.rglob("*.json"))
        
        if not files:
            print(f"⚠️ No processed files found in {base_dir}. Check your ingestion!", flush=True)
            return

        print(f"📂 Found {len(files)} file(s). Starting graph construction...", flush=True)
        
        with self.driver.session() as session:
            for file_path in files:
                print(f"📄 Processing: {file_path.name}", flush=True)
                with open(file_path, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                    
                for i, chunk in enumerate(chunks):
                    print(f"   ⚡ Chunk {i+1}/{len(chunks)}...", flush=True)
                    # Hierarchical Metadata
                    session.run("""
                        MERGE (c:Company {name: $company})
                        MERGE (q:Quarter {name: $quarter})
                        MERGE (c)-[:HAS_QUARTER]->(q)
                        MERGE (chunk:TextChunk {content: $text})
                        SET chunk.source = $source, chunk.type = $type
                        MERGE (q)-[:HAS_DATA]->(chunk)
                    """, company=chunk['company'], quarter=chunk['quarter'], 
                         text=chunk['text'], source=chunk['source'], type=chunk['type'])

                    graph_data = self._extract_graph_data(chunk['text'])
                    
                    for ent in graph_data.entities:
                        session.run(f"""
                            MERGE (e:{ent.type} {{name: $name}})
                            ON CREATE SET e.value = $value
                            WITH e
                            MATCH (chunk:TextChunk {{content: $text}})
                            MERGE (chunk)-[:MENTIONS]->(e)
                        """, name=ent.name, value=ent.value, text=chunk['text'])

                    for rel in graph_data.relationships:
                        rel_type = rel.relation.upper().replace(' ', '_')
                        session.run(f"""
                            MATCH (a {{name: $source}}), (b {{name: $target}})
                            MERGE (a)-[:{rel_type}]->(b)
                        """, source=rel.source, target=rel.target)

    def close(self):
        self.driver.close()


def clean_json(text):
    """Simple helper to ensure JSON is valid before parsing."""
    return text.strip().replace("```json", "").replace("```", "")

if __name__ == "__main__":
    builder = GraphBuilder()
    builder.build_graph_from_files()
    builder.close()
    print("✅ Full Graph Construction Complete.", flush=True)