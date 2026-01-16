import os
import ollama
from neo4j import GraphDatabase

class Emb:
    def __init__(self):
        host = os.getenv("OLLAMA_HOST","http://ollama:11434")
        self.client = ollama.Client(host=host)
        self.model = os.getenv("OLLAMA_EMBED_MODEL","nomic-embed-text")

    def encode(self, text):
        return self.client.embeddings(model=self.model, prompt=text)["embedding"]

class Neo:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
        )

    def close(self):
        self.driver.close()

    def create_index(self, dim):
        with self.driver.session() as s:
            s.run("DROP INDEX KG_CHUNK_VECTOR IF EXISTS;")
            s.run(
                """
                CALL db.index.vector.createNodeIndex(
                    'KG_CHUNK_VECTOR',
                    'Chunk',
                    'embedding',
                    $dim,
                    'cosine'
                )
                """,
                dim=dim
            )

    def update_chunk(self, cid, emb):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (c:Chunk {id:$id})
                SET c.embedding = $v
                """,
                id=cid,
                v=emb
            )

def main():
    emb = Emb()
    neo = Neo()

    with neo.driver.session() as s:
        recs = s.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text")
        rows = list(recs)

    if not rows:
        neo.close()
        return

    d = emb.encode("dimension probe text")
    dim = len(d)

    neo.create_index(dim)

    for r in rows:
        e = emb.encode(r["text"])
        neo.update_chunk(r["id"], e)

    neo.close()

if __name__ == "__main__":
    main()
