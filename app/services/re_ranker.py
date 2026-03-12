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
    
    avg_rerank_score = np.mean([c["rerank_score"] for c in chunks])
    
    if avg_rerank_score < 0.1:
        sorted_chunks = sorted(chunks, key=lambda x: x.get("vector_score", 0), reverse=True)
        for chunk in sorted_chunks:
            chunk["rerank_fallback"] = True
    else:
        sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        for chunk in sorted_chunks:
            chunk["rerank_fallback"] = False
    
    return sorted_chunks[:top_k]