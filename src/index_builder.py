import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

class IndexBuilder:
    def __init__(self):
        # 1. Initialize the Embedding Model (Free, runs locally)
        # This model turns text into a 384-dimensional vector.
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 2. Initialize Neo4j Connection
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        self.index_name = "text-chunk-embeddings"

    def create_vector_index(self):
        """Initializes the Vector Index in Neo4j."""
        print(f"🛠️ Creating Vector Index: {self.index_name}...")
        with self.driver.session() as session:
            # We create a vector index on the 'embedding' property of 'TextChunk' nodes
            session.run(f"""
                CREATE VECTOR INDEX `{self.index_name}` IF NOT EXISTS
                FOR (n:TextChunk) ON (n.embedding)
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
        print("✅ Vector Index initialized.")

    def populate_embeddings(self):
        """Finds chunks without embeddings and generates them."""
        print("🧠 Generating embeddings for TextChunks...")
        with self.driver.session() as session:
            # Find chunks that don't have an embedding yet
            result = session.run("MATCH (n:TextChunk) WHERE n.embedding IS NULL RETURN n.content AS text, id(n) AS id")
            records = list(result)
            
            if not records:
                print("✨ All chunks already have embeddings.")
                return

            for record in records:
                text = record['text']
                node_id = record['id']
                
                # Generate the vector
                vector = self.model.encode(text).tolist()
                
                # Save the vector back to Neo4j
                session.run("""
                    MATCH (n) WHERE id(n) = $id
                    CALL db.create.setNodeVectorProperty(n, 'embedding', $vector)
                """, id=node_id, vector=vector)
                
            print(f"✅ Successfully updated {len(records)} chunks.")

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    builder = IndexBuilder()
    builder.create_vector_index()
    builder.populate_embeddings()
    builder.close()