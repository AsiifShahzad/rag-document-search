from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

from app.models.schemas import AskRequest
from app.rag.pipeline import retrieve_context
from app.rag.retriever import query_embedding, retrieve_chunks
from app.rag.generator import generate_answer
from app.services.prompt_builder import build_prompt
from app.services.response_builder import format_response
from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search
from app.services.re_ranker import get_reranker
from app.rag.pipeline import retrieve_context
from app.rag.generator import generate_answer
from app.services.ingestion_pipeline import data_ingestion

router = APIRouter()

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/health")
async def health_check():
    status = {}

    try:
        model = get_embedding_model()
        test_vector = model.embed_query("health check")
        status["embedding_model"] = "ok" if len(test_vector) == 384 else "wrong dimension"
    except Exception as e:
        status["embedding_model"] = f"failed — {str(e)}"

    try:
        dummy_vector = [0.0] * 384
        result = similarity_search(dummy_vector, top_k=1)
        status["pinecone"] = "ok"
    except Exception as e:
        status["pinecone"] = f"failed — {str(e)}"

    try:
        reranker = get_reranker()
        test_score = reranker.predict([("test query", "test chunk")])
        status["reranker"] = "ok"
    except Exception as e:
        status["reranker"] = f"failed — {str(e)}"

    all_ok = all(v == "ok" for v in status.values())
    status["overall"] = "healthy" if all_ok else "degraded"

    return status


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")

    storage_path = UPLOAD_DIR / file.filename

    with storage_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks, embeddings = data_ingestion(str(storage_path))

    return JSONResponse(
        content={
            "filename": file.filename,
            "chunks_processed": len(chunks)
        }
    )

@router.post("/ask")
async def ask_question(req: AskRequest):

    query = req.question.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve_context(query)

    if not chunks:
        return {
            "answer": "No relevant information found",
            "sources": [],
            "confidence": 0.0
        }

    prompt = build_prompt(chunks, query)

    answer = generate_answer(prompt)
    rerank_scores = [c.get("rerank_score", 0.0) for c in chunks]
    response = format_response(answer, chunks, rerank_scores)

    return response