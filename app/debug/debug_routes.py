"""
Debug API endpoints for RAG pipeline verification
Include this router in your main app for comprehensive debugging
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import json

from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search, index as pinecone_index, INDEX_NAME
from app.services.re_ranker import get_reranker
from app.rag.pipeline import retrieve_context

debug_router = APIRouter(prefix="/debug", tags=["debug"])


class DebugResponse:
    """Helper class to build structured debug responses"""
    def __init__(self):
        self.data = {
            "status": "ok",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "results": {}
        }
    
    def set_error(self, error: str):
        self.data["status"] = "error"
        self.data["error"] = error
        return self.data
    
    def add_result(self, key: str, value):
        self.data["results"][key] = value
        return self
    
    def get(self):
        return self.data


@debug_router.get("/health-detailed")
async def debug_health_check():
    """
    Detailed health check with component diagnostics
    """
    response = DebugResponse()
    components = {}
    
    # Check embedding model
    try:
        model = get_embedding_model()
        test_vector = model.embed_query("health check")
        test_array = np.array(test_vector)
        components["embedding_model"] = {
            "status": "ok",
            "dimensions": int(test_array.shape[0]),
            "norm": float(np.linalg.norm(test_array)),
            "has_zeros": bool(np.all(test_array == 0)),
            "has_nans": bool(np.any(np.isnan(test_array)))
        }
    except Exception as e:
        components["embedding_model"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check Pinecone
    try:
        dummy_vector = [0.0] * 384
        result = similarity_search(dummy_vector, top_k=1)
        stats = pinecone_index.describe_index_stats()
        components["pinecone"] = {
            "status": "ok",
            "total_vectors": stats.total_vector_count,
            "index_name": INDEX_NAME,
            "connection": "active"
        }
    except Exception as e:
        components["pinecone"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check reranker
    try:
        reranker = get_reranker()
        test_pairs = [("test", "test")]
        test_scores = reranker.predict(test_pairs)
        components["reranker"] = {
            "status": "ok",
            "model": "BAAI/bge-reranker-base",
            "test_score": float(test_scores[0])
        }
    except Exception as e:
        components["reranker"] = {
            "status": "error",
            "error": str(e)
        }
    
    overall = "ok" if all(c.get("status") == "ok" for c in components.values()) else "degraded"
    
    return response.add_result("components", components).add_result("overall", overall).get()


@debug_router.get("/pinecone-stats")
async def debug_pinecone_stats():
    """
    Get detailed Pinecone index statistics
    """
    response = DebugResponse()
    try:
        stats = pinecone_index.describe_index_stats()
        
        stats_data = {
            "total_vectors": stats.total_vector_count,
            "index_name": INDEX_NAME,
            "dimension": 384,
            "namespaces": stats.namespaces if hasattr(stats, 'namespaces') else "default"
        }
        
        # Get sample vectors
        dummy_vector = [0.0] * 384
        sample_results = similarity_search(dummy_vector, top_k=min(5, stats.total_vector_count))
        
        sample_vectors = []
        for match in sample_results.get("matches", []):
            metadata = match.get("metadata", {})
            sample_vectors.append({
                "id": match.get("id"),
                "score": float(match.get("score")),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "text_preview": metadata.get("text", "")[:100]
            })
        
        stats_data["sample_vectors"] = sample_vectors
        
        return response.add_result("pinecone_stats", stats_data).get()
    
    except Exception as e:
        return response.set_error(f"Pinecone stats failed: {str(e)}")


@debug_router.get("/test-embedding")
async def debug_test_embedding(query: str = Query(...)):
    """
    Test embedding generation for a query
    """
    response = DebugResponse()
    try:
        model = get_embedding_model()
        embedding = model.embed_query(query)
        embedding_array = np.array(embedding)
        
        embedding_data = {
            "query": query,
            "dimensions": int(embedding_array.shape[0]),
            "norm": float(np.linalg.norm(embedding_array)),
            "mean": float(embedding_array.mean()),
            "std": float(embedding_array.std()),
            "min": float(embedding_array.min()),
            "max": float(embedding_array.max()),
            "sample_values": embedding_array[:10].tolist()
        }
        
        return response.add_result("embedding", embedding_data).get()
    
    except Exception as e:
        return response.set_error(f"Embedding test failed: {str(e)}")


@debug_router.get("/test-retrieval")
async def debug_test_retrieval(
    query: str = Query(...),
    top_k: int = Query(10, ge=1, le=50)
):
    """
    Test vector similarity search and retrieval
    """
    response = DebugResponse()
    try:
        model = get_embedding_model()
        query_vector = model.embed_query(query)
        query_array = np.array(query_vector)
        
        results = similarity_search(query_array.tolist(), top_k=top_k)
        matches = results.get("matches", [])
        
        scores = [float(m.get("score", 0)) for m in matches]
        
        retrieved_chunks = []
        for i, match in enumerate(matches):
            metadata = match.get("metadata", {})
            retrieved_chunks.append({
                "rank": i + 1,
                "id": match.get("id"),
                "score": float(match.get("score", 0)),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "text_preview": metadata.get("text", "")[:150]
            })
        
        retrieval_data = {
            "query": query,
            "results_count": len(matches),
            "avg_score": float(np.mean(scores)) if scores else 0,
            "max_score": float(np.max(scores)) if scores else 0,
            "min_score": float(np.min(scores)) if scores else 0,
            "chunks": retrieved_chunks
        }
        
        return response.add_result("retrieval", retrieval_data).get()
    
    except Exception as e:
        return response.set_error(f"Retrieval test failed: {str(e)}")


@debug_router.get("/test-reranking")
async def debug_test_reranking(
    query: str = Query(...),
    top_k: int = Query(5, ge=1, le=20)
):
    """
    Test full retrieval and reranking pipeline
    """
    response = DebugResponse()
    try:
        # Get raw retrieval results
        model = get_embedding_model()
        query_vector = model.embed_query(query)
        query_array = np.array(query_vector)
        
        raw_results = similarity_search(query_array.tolist(), top_k=20)
        raw_matches = raw_results.get("matches", [])
        
        # Convert to chunk format
        chunks = []
        for match in raw_matches:
            metadata = match.get("metadata", {})
            chunks.append({
                "text": metadata.get("text", ""),
                "source": metadata.get("source"),
                "page": metadata.get("page"),
                "vector_score": float(match.get("score", 0))
            })
        
        # Rerank
        if not chunks:
            return response.set_error("No vectors found for reranking")
        
        from app.services.re_ranker import rerank_chunks
        reranked = rerank_chunks(query, chunks, top_k=top_k)
        
        # Format results
        reranking_data = {
            "query": query,
            "raw_retrieval_count": len(chunks),
            "reranked_count": len(reranked),
            "ranked_chunks": []
        }
        
        for i, chunk in enumerate(reranked):
            reranking_data["ranked_chunks"].append({
                "rank": i + 1,
                "rerank_score": float(chunk.get("rerank_score", 0)),
                "vector_score": float(chunk.get("vector_score", 0)),
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "text_preview": chunk.get("text", "")[:150]
            })
        
        return response.add_result("reranking", reranking_data).get()
    
    except Exception as e:
        return response.set_error(f"Reranking test failed: {str(e)}")


@debug_router.get("/test-pipeline")
async def debug_test_full_pipeline(query: str = Query(...)):
    """
    Test complete RAG pipeline from query to retrieval
    """
    response = DebugResponse()
    try:
        print(f"\n[DEBUG ENDPOINT] Testing full pipeline for query: '{query}'")
        
        chunks = retrieve_context(query)
        
        pipeline_data = {
            "query": query,
            "chunks_retrieved": len(chunks),
            "chunks": []
        }
        
        if not chunks:
            pipeline_data["status"] = "warning"
            pipeline_data["message"] = "No chunks retrieved from pipeline"
        else:
            for i, chunk in enumerate(chunks):
                pipeline_data["chunks"].append({
                    "rank": i + 1,
                    "rerank_score": float(chunk.get("rerank_score", 0)),
                    "vector_score": float(chunk.get("vector_score", 0)),
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "text_length": len(chunk.get("text", "")),
                    "text_preview": chunk.get("text", "")[:150]
                })
        
        return response.add_result("pipeline", pipeline_data).get()
    
    except Exception as e:
        return response.set_error(f"Pipeline test failed: {str(e)}")


@debug_router.get("/verify-all")
async def debug_verify_all(test_query: Optional[str] = Query(None)):
    """
    Run complete verification of all pipeline components
    Returns comprehensive diagnostic report
    """
    response = DebugResponse()
    
    verification = {
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Embedding Model
    try:
        model = get_embedding_model()
        test_vector = model.embed_query("test")
        test_array = np.array(test_vector)
        verification["tests"]["embedding_model"] = {
            "status": "✓ Pass",
            "dimensions": int(test_array.shape[0]),
            "norm": float(np.linalg.norm(test_array))
        }
    except Exception as e:
        verification["tests"]["embedding_model"] = {
            "status": "✗ Fail",
            "error": str(e)
        }
    
    # Test 2: Pinecone Connection
    try:
        stats = pinecone_index.describe_index_stats()
        verification["tests"]["pinecone_connection"] = {
            "status": "✓ Pass",
            "total_vectors": stats.total_vector_count
        }
    except Exception as e:
        verification["tests"]["pinecone_connection"] = {
            "status": "✗ Fail",
            "error": str(e)
        }
    
    # Test 3: Reranker
    try:
        reranker = get_reranker()
        scores = reranker.predict([("test", "test")])
        verification["tests"]["reranker"] = {
            "status": "✓ Pass",
            "model": "BAAI/bge-reranker-base"
        }
    except Exception as e:
        verification["tests"]["reranker"] = {
            "status": "✗ Fail",
            "error": str(e)
        }
    
    # Test 4: End-to-end (if query provided)
    if test_query:
        try:
            chunks = retrieve_context(test_query)
            verification["tests"]["end_to_end"] = {
                "status": "✓ Pass" if chunks else "⚠ Warning",
                "query": test_query,
                "chunks_retrieved": len(chunks),
                "message": "Pipeline works" if chunks else "Query returned no results"
            }
        except Exception as e:
            verification["tests"]["end_to_end"] = {
                "status": "✗ Fail",
                "error": str(e)
            }
    
    # Overall assessment
    failed_tests = [t for t, v in verification["tests"].items() if "Fail" in v.get("status", "")]
    verification["overall_status"] = "Healthy" if not failed_tests else f"{len(failed_tests)} test(s) failed"
    
    return response.add_result("verification", verification).get()


@debug_router.get("/logs-sample")
async def debug_logs_sample():
    """
    Returns information about recent operations and logs
    Note: For full logs, check your application's log output
    """
    response = DebugResponse()
    
    logs_info = {
        "note": "Full logs are printed to your terminal/console",
        "features": [
            "Document ingestion logs with chunk details",
            "Embedding generation logs with vector statistics",
            "Vector search logs with similarity scores",
            "Reranking logs with score comparisons",
            "LLM prompt and response logs",
            "Phase-by-phase debug output"
        ],
        "to_view_logs": [
            "Watch the backend terminal/console output",
            "Each /ask request will print detailed pipeline information",
            "Each /upload request will print ingestion details",
            "Use /debug endpoints to see structured diagnostic data"
        ]
    }
    
    return response.add_result("logs", logs_info).get()
