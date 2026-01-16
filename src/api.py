from fastapi import FastAPI
from pydantic import BaseModel
from src.retrieval_and_answer import run_query

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query_endpoint(payload: QueryRequest):
    answer = run_query(payload.query)
    return {"answer": answer}
