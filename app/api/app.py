from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import shutil

from ingestion_pipeline import data_ingestion
from retriever import query_embedding, retrieve_chunks
from prompt_builder import build_prompt
from generator import generate_answer
from response_builder import format_response


app = FastAPI(title="RAG Backend")

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    question: str


@app.post("/upload")
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


@app.post("/ask")
async def ask_question(req: AskRequest):

    query = req.question.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_vector = embed_query(query)

    chunks, scores = retrieve_chunks(query_vector)

    if not chunks:
        return {
            "answer": "No relevant information found",
            "sources": [],
            "confidence": 0.0
        }

    prompt = build_prompt(chunks, query)

    answer = generate_answer(prompt)

    response = format_response(answer, chunks, scores)

    return response