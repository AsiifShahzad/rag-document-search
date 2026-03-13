from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil

from app.models.schemas import AskRequest
from app.rag.pipeline import retrieve_context
from app.rag.generator import generate_answer
from app.services.prompt_builder import build_prompt
from app.services.response_builder import format_response
from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search, delete_document_embeddings, delete_session_embeddings
from app.services.re_ranker import get_reranker
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
async def upload_pdf(file: UploadFile = File(...), session_id: str = None):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF allowed")
    
    if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
        raise HTTPException(status_code=400, detail="Valid session_id is required")

    # Validate session_id format (alphanumeric, underscore, dash only)
    import re
    if not re.match(r'^[a-zA-Z0-9_-]{10,100}$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    storage_path = UPLOAD_DIR / file.filename

    try:
        with storage_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks, embeddings = data_ingestion(str(storage_path), session_id)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "filename": file.filename,
                "chunks_processed": len(chunks),
                "session_id": session_id
            }
        )
    except ValueError as e:
        # Clean up file if ingestion fails
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except RuntimeError as e:
        # Clean up file if embedding fails
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except Exception as e:
        # Clean up file on any unexpected error
        if storage_path.exists():
            storage_path.unlink()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

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

@router.delete("/delete/{document_name}")
async def delete_document(document_name: str):
    """Delete a document and all its embeddings from Pinecone"""
    
    # Ensure PDF extension
    if not document_name.lower().endswith(".pdf"):
        document_name = f"{document_name}.pdf"
    
    # Delete from disk if it exists
    file_path = UPLOAD_DIR / document_name
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Could not delete file from disk: {str(e)}"
                }
            )
    
    # Delete embeddings from Pinecone
    result = delete_document_embeddings(document_name)
    
    if result["success"]:
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Document '{document_name}' deleted successfully",
                "embeddings_removed": result.get("deleted_count", 0)
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Error deleting embeddings: {result.get('error', 'Unknown error')}"
            }
        )

@router.post("/cleanup-session/{session_id}")
async def cleanup_session(session_id: str):
    """Delete all embeddings for a session (called when user leaves/refreshes)"""
    
    # Validate session_id format
    import re
    if not session_id or not re.match(r'^[a-zA-Z0-9_-]{10,100}$', session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")
    
    try:
        result = delete_session_embeddings(session_id)
        
        if result["success"]:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"Session cleaned up successfully",
                    "embeddings_removed": result.get("deleted_count", 0)
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": f"Error cleaning up session: {result.get('error', 'Unknown error')}"
                }
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Unexpected error during cleanup: {str(e)}"
            }
        )