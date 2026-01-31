import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

load_dotenv()

class RetrievalEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Performs Multi-Hop Retrieval:
        1. Vector search for the 'Entry' TextChunks.
        2. Graph traversal to find Metrics, Quarters, and Entity relationships.
        """
        query_vector = self.model.encode(query).tolist()
        
        with self.driver.session() as session:
            # Cypher Query: Find chunks, then 'hop' to their metadata and entities
            cypher = """
            CALL db.index.vector.queryNodes('text-chunk-embeddings', $k, $vector)
            YIELD node AS chunk, score
            
            // Hop 1: Get Document Metadata
            MATCH (q:Quarter)-[:HAS_DATA]->(chunk)
            MATCH (comp:Company)-[:HAS_QUARTER]->(q)
            
            // Hop 2: Get Connected Entities & Metrics
            OPTIONAL MATCH (chunk)-[:MENTIONS]->(e)
            OPTIONAL MATCH (e)-[rel]-(neighbor)
            WHERE NOT neighbor:TextChunk AND NOT neighbor:Quarter
            
            RETURN 
                chunk.content AS text,
                chunk.source AS source,
                comp.name AS company,
                q.name AS quarter,
                collect(DISTINCT {
                    entity: e.name, 
                    type: labels(e)[0], 
                    relation: type(rel), 
                    target: neighbor.name,
                    val: e.value
                }) AS graph_knowledge
            """
            result = session.run(cypher, vector=query_vector, k=top_k)
            
            context_blocks = []
            for record in result:
                block = f"--- SOURCE: {record['company']} {record['quarter']} ({record['source']}) ---\n"
                block += f"TEXT: {record['text']}\n"
                
                # Format the 'Hops' found for this chunk
                if record['graph_knowledge']:
                    knowledge = []
                    for k in record['graph_knowledge']:
                        if k['entity']:
                            fact = f"- {k['entity']} ({k['type']})"
                            if k['val']: fact += f" Value: {k['val']}"
                            if k['relation']: fact += f" [{k['relation']} -> {k['target']}]"
                            knowledge.append(fact)
                    block += "GRAPH INSIGHTS:\n" + "\n".join(list(set(knowledge))[:10]) # Deduplicate
                
                context_blocks.append(block)
            
            return "\n\n".join(context_blocks)

    def close(self):
        self.driver.close()