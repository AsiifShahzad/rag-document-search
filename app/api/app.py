import logging
import sys
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
import traceback

from app.models.schemas import AskRequest
from app.rag.pipeline import retrieve_context
from app.rag.generator import generate_answer
from app.services.prompt_builder import build_prompt
from app.services.response_builder import format_response
from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search, delete_document_embeddings, delete_session_embeddings
from app.services.re_ranker import get_reranker
from app.services.ingestion_pipeline import data_ingestion

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = Path("data/documents")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logger.info(f"Upload directory: {UPLOAD_DIR}")


@router.get("/health")
async def health_check():
    """Health check endpoint that verifies all dependencies"""
    logger.info("Health check requested")
    status = {}

    try:
        model = get_embedding_model()
        test_vector = model.embed_query("health check")
        status["embedding_model"] = "ok" if len(test_vector) == 384 else "wrong dimension"
        logger.info(f"Embedding model: {status['embedding_model']}")
    except Exception as e:
        error_msg = f"failed - {str(e)}"
        status["embedding_model"] = error_msg
        logger.error(f"Embedding model check failed: {error_msg}\n{traceback.format_exc()}")

    try:
        dummy_vector = [0.0] * 384
        result = similarity_search(dummy_vector, top_k=1)
        status["pinecone"] = "ok"
        logger.info("Pinecone vector store: ok")
    except Exception as e:
        error_msg = f"failed - {str(e)}"
        status["pinecone"] = error_msg
        logger.error(f"Pinecone check failed: {error_msg}\n{traceback.format_exc()}")

    try:
        reranker = get_reranker()
        test_score = reranker.predict([("test query", "test chunk")])
        status["reranker"] = "ok"
        logger.info("Reranker: ok")
    except Exception as e:
        error_msg = f"failed - {str(e)}"
        status["reranker"] = error_msg
        logger.error(f"Reranker check failed: {error_msg}\n{traceback.format_exc()}")

    all_ok = all(v == "ok" for v in status.values())
    status["overall"] = "healthy" if all_ok else "degraded"
    
    http_status = 200 if all_ok else 503
    logger.info(f"Health check result: {status['overall']} (HTTP {http_status})")
    
    return JSONResponse(status_code=http_status, content=status)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), session_id: str = Form(None)):
    """
    Upload a PDF document for processing.
    Returns session_id and number of chunks processed.
    """
    logger.info(f"Upload request: filename={file.filename}, session_id={session_id}")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith(".pdf"):
            logger.warning(f"Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Auto-generate session_id if not provided
        if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session_id: {session_id}")
        
        # Clean up previous embeddings for this session
        try:
            delete_session_embeddings(session_id)
            logger.info(f"Cleaned up previous embeddings for session: {session_id}")
        except Exception as e:
            logger.warning(f"Could not clean previous embeddings: {str(e)}")

        # Save file to disk
        storage_path = UPLOAD_DIR / file.filename
        logger.info(f"Saving file to: {storage_path}")
        
        try:
            with storage_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved successfully: {storage_path}")
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # Process document through ingestion pipeline
        logger.info(f"Starting data ingestion for: {file.filename}")
        try:
            chunks, embeddings = data_ingestion(str(storage_path), session_id)
            logger.info(f"Data ingestion completed: {len(chunks)} chunks, {len(embeddings)} embeddings")
        except ValueError as e:
            logger.error(f"Validation error during ingestion: {str(e)}\n{traceback.format_exc()}")
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
        except RuntimeError as e:
            logger.error(f"Runtime error during ingestion: {str(e)}\n{traceback.format_exc()}")
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {str(e)}\n{traceback.format_exc()}")
            if storage_path.exists():
                storage_path.unlink()
            raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

        response_data = {
            "success": True,
            "filename": file.filename,
            "chunks_processed": len(chunks),
            "session_id": session_id
        }
        logger.info(f"Upload successful: {response_data}")
        
        return JSONResponse(status_code=200, content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/cleanup")
async def cleanup_session(session_id: str = Form(...)):
    """Delete all embeddings for a session when user leaves/refreshes"""
    logger.info(f"Cleanup requested for session: {session_id}")
    
    if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
        logger.warning("Cleanup request with invalid session_id")
        raise HTTPException(status_code=400, detail="Valid session_id is required")
    
    try:
        result = delete_session_embeddings(session_id)
        if result["success"]:
            response_data = {
                "success": True,
                "message": f"Session {session_id} cleaned up",
                "deleted_count": result.get("deleted_count", 0)
            }
            logger.info(f"Cleanup successful: {response_data}")
            return JSONResponse(status_code=200, content=response_data)
        else:
            error_msg = result.get("error", "Cleanup failed")
            logger.error(f"Cleanup failed: {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Cleanup error: {str(e)}")

@router.post("/ask")
async def ask_question(req: AskRequest):
    """Process user question and return answer with sources"""
    logger.info(f"Question received from session {req.session_id}: {req.question[:100]}...")
    
    try:
        query = req.question.strip()

        if not query:
            logger.warning("Empty question received")
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        if not req.session_id:
            logger.warning("Question without session_id")
            raise HTTPException(status_code=400, detail="session_id is required")
        
        # Retrieve context
        logger.info(f"Retrieving context for query: {query[:50]}...")
        chunks = retrieve_context(query, session_id=req.session_id)
        logger.info(f"Retrieved {len(chunks) if chunks else 0} chunks")

        if not chunks:
            logger.info("No relevant chunks found, returning empty response")
            return {
                "answer": "No relevant information found in the uploaded documents.",
                "sources": [],
                "confidence": 0.0
            }

        # Generate answer
        logger.info("Building prompt and generating answer...")
        prompt = build_prompt(chunks, query)
        answer = generate_answer(prompt)
        logger.info(f"Answer generated: {answer[:100]}...")
        
        # Format response with rerank scores
        rerank_scores = [c.get("rerank_score", 0.0) for c in chunks]
        response = format_response(answer, chunks, rerank_scores)
        
        logger.info(f"Response formatted with {len(chunks)} sources")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
@router.get("/debug-session/{session_id}")
async def debug_session(session_id: str):
    """Temporary debug endpoint - remove after fixing"""
    from app.services.vector_store import index, DIMENSION
    
    # Check what vectors exist for this session
    results = index.query(
        vector=[0.0] * DIMENSION,
        top_k=5,
        include_metadata=True,
        filter={"session_id": {"$eq": session_id}}
    )
    
    # Also check WITHOUT filter to see if ANY vectors exist at all
    results_no_filter = index.query(
        vector=[0.0] * DIMENSION,
        top_k=5,
        include_metadata=True
    )
    
    return {
        "session_id": session_id,
        "vectors_with_filter": len(results.get("matches", [])),
        "sample_filtered": [m["metadata"] for m in results.get("matches", [])[:2]],
        "total_vectors_no_filter": len(results_no_filter.get("matches", [])),
        "sample_any": [m["metadata"] for m in results_no_filter.get("matches", [])[:2]],
    }

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