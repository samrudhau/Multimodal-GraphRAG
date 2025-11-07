
import os
import json
from pathlib import Path
import ollama # <-- New import
from pydantic import BaseModel, Field
from typing import List

from neo4j import GraphDatabase, Driver

# --- Pydantic Models for Structured Data (Our Schema in Code) ---
# --- (This section is unchanged) ---

class Speaker(BaseModel):
    name: str = Field(description="The name of the person speaking.")
    title: str = Field(description="The title of the speaker, e.g., 'CEO'.")

class Organization(BaseModel):
    name: str = Field(description="The name of the company or organization.")

class Product(BaseModel):
    name: str = Field(description="The name of a product, technology, or service.")

class KeyConcept(BaseModel):
    name: str = Field(description="An important theme, strategy, or topic discussed.")
    
class FinancialMetric(BaseModel):
    name: str = Field(description="A specific financial term or data point mentioned.")

Node = Speaker | Organization | Product | KeyConcept | FinancialMetric

class Relationship(BaseModel):
    source: str = Field(description="Name of the source node.")
    target: str = Field(description="Name of the target node.")
    type: str = Field(description="The type of the relationship between source and target.")

class KnowledgeGraph(BaseModel):
    nodes: List[Node] = Field(description="List of all entities (nodes) extracted from the text.")
    relationships: List[Relationship] = Field(description="List of all relationships between the entities.")


# --- Ollama Knowledge Extractor (Replaces OpenAI version) ---

class OllamaKnowledgeGraphExtractor:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

        # NEW: Read Ollama host from environment variable, fallback to localhost
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.client = ollama.Client(host=ollama_host)
        print(f"Initialized Ollama client with model '{self.model_name}' at {ollama_host}")
    
    def _generate_prompt(self, text: str):
        # We provide the Pydantic schema and one example to guide the model.
        # This is a form of "one-shot" prompting.
        return f"""
        You are a top-tier algorithm for extracting information in JSON format.
        
        Extract all entities and relationships from the following text based on the provided schema.
        
        Respond with a JSON object that strictly follows this Pydantic schema:
        {KnowledgeGraph.model_json_schema()}
        
        Here is an example of the expected output format:
        {{
          "nodes": [
            {{
              "name": "Jensen Huang",
              "title": "CEO"
            }},
            {{
              "name": "NVIDIA"
            }}
          ],
          "relationships": [
            {{
              "source": "Jensen Huang",
              "target": "NVIDIA",
              "type": "IS_EXECUTIVE_OF"
            }}
          ]
        }}
        
        Here is the text to analyze:
        ---
        {text}
        ---
        """

    def extract(self, text: str) -> KnowledgeGraph:
        print("Extracting knowledge from text chunk using Ollama...")
        try:
            prompt = self._generate_prompt(text)
            
            response = self.client.chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' # This tells Ollama to guarantee a JSON output
            )
            
            # The response content is a JSON string, so we parse it
            json_response = json.loads(response['message']['content'])
            graph = KnowledgeGraph.model_validate(json_response)
            
            print(f"Extracted {len(graph.nodes)} nodes and {len(graph.relationships)} relationships.")
            return graph
        except Exception as e:
            print(f"Error during Ollama extraction: {e}")
            return KnowledgeGraph(nodes=[], relationships=[])


# --- Neo4j Database Interaction ---
# --- (This section is unchanged) ---
class Neo4jDatabase:
    def __init__(self, uri, user, password):
        self.driver: Driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def import_graph(self, graph: KnowledgeGraph):
        print("Importing graph into Neo4j...")
        with self.driver.session() as session:
            for node in graph.nodes:
                node_label = node.__class__.__name__
                node_properties = node.model_dump()
                query = f"MERGE (n:{node_label} {{name: $name}}) SET n += $props"
                session.run(query, name=node.name, props=node_properties)

            for rel in graph.relationships:
                query = f"""
                MATCH (a {{name: $source_name}}), (b {{name: $target_name}})
                MERGE (a)-[r:{rel.type}]->(b)
                """
                session.run(query, source_name=rel.source, target_name=rel.target)
        print("Graph import complete.")


def main():
    # --- CONFIGURATION ---
    PROCESSED_JSON_FILENAME = "NVDA-Q1-2025_processed.json"
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_neo4j_password")
    # ---------------------

    base_path = Path(__file__).resolve().parent.parent
    processed_file_path = base_path / "data" / "processed" / PROCESSED_JSON_FILENAME
    
    with open(processed_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    segments = data['whisper_transcription']['segments']
    
    CHUNK_SIZE = 10 
    text_chunks = []
    current_chunk = ""
    for i, segment in enumerate(segments):
        current_chunk += segment['text'] + " "
        if (i + 1) % CHUNK_SIZE == 0 or (i + 1) == len(segments):
            text_chunks.append(current_chunk.strip())
            current_chunk = ""
    
    print(f"Divided transcript into {len(text_chunks)} chunks.")

    # Initialize tools with the new Ollama extractor
    extractor = OllamaKnowledgeGraphExtractor() # <-- THE ONLY CHANGE IN THIS SECTION
    db = Neo4jDatabase(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    # Process each chunk and import into Neo4j
    # We can now process more chunks since it's free! Let's do 10.
    for i, chunk in enumerate(text_chunks[:10]):
        print(f"\n--- Processing Chunk {i+1}/{len(text_chunks[:10])} ---")
        extracted_graph = extractor.extract(chunk)
        if extracted_graph.nodes:
            db.import_graph(extracted_graph)
    
    db.close()
    print("\nAll chunks processed. Knowledge graph construction is complete.")


if __name__ == "__main__":
    main()