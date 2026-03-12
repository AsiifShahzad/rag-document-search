from sentence_transformers import CrossEncoder
from typing import List, Dict
import numpy as np

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("BAAI/bge-reranker-base")
    return _reranker

def rerank_chunks(query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Rerank chunks by relevance to query.
    
    IMPORTANT: If reranking scores collapse (avg < 0.1), falls back to vector search scores
    because the reranker model may not understand the query semantics.
    """
    reranker = get_reranker()
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)
    
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    
    # Check if reranking collapsed
    avg_rerank_score = np.mean([c["rerank_score"] for c in chunks])
    
    print(f"\n[RERANKER] Average rerank score: {avg_rerank_score:.4f}")
    
    if avg_rerank_score < 0.1:
        # Reranker scores are critically low - indicates poor semantic match
        # Fall back to vector search scores which performed well
        print(f"[RERANKER] ⚠️  Low avg score detected. Falling back to vector search scores.")
        print(f"[RERANKER] This typically means: query is indirect/generic, or semantically distant from document")
        print(f"[RERANKER] Using vector similarity scores instead of rerank scores...")
        
        sorted_chunks = sorted(chunks, key=lambda x: x.get("vector_score", 0), reverse=True)
        
        # Mark that we used fallback
        for chunk in sorted_chunks:
            chunk["rerank_fallback"] = True
    else:
        # Normal reranking worked fine
        print(f"[RERANKER] ✓ Reranking scores healthy. Using reranked order.")
        sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        for chunk in sorted_chunks:
            chunk["rerank_fallback"] = False
    
    # Print ranking details
    for i, chunk in enumerate(sorted_chunks[:top_k]):
        method = "fallback (vector)" if chunk.get("rerank_fallback") else "rerank"
        print(f"  [{i+1}] Score: {chunk.get('rerank_score', chunk.get('vector_score', 0)):.4f} ({method})")
    
    return sorted_chunks[:top_k]