from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval import RetrievalEngine
from src.generation import GenerationEngine

app = FastAPI(title="NVIDIA GraphRAG API")

# Initialize our specialized engines
retriever = RetrievalEngine()
generator = GenerationEngine()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        # 1. Retrieval: Navigation through the Neo4j Graph
        context = retriever.search(request.question)
        
        if not context:
            return {
                "question": request.question,
                "answer": "I couldn't find any relevant information in the knowledge graph.",
                "sources": "None"
            }

        # 2. Generation: Synthesizing the Graph Insights into an answer
        # Note: Changed to .answer() to match our GenerationEngine class
        answer = generator.answer(request.question, context)
        
        return {
            "question": request.question,
            "answer": answer,
            "sources": context # This maps to the 'expander' in your Streamlit app
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    retriever.close()