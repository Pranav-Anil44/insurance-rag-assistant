from fastapi import APIRouter
from app.models.schema import QueryRequest
from src.rag_pipeline import ask_rag

router = APIRouter()

@router.post("/ask")
def ask_question(request: QueryRequest):
    result = ask_rag(request.question)
    return result