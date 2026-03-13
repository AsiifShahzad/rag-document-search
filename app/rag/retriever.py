from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search
from app.services.re_ranker import rerank_chunks
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def query_embedding(query: str) -> np.ndarray:
    model = get_embedding_model()
    vector = model.embed_query(query)
    return np.array(vector)

def retrieve_chunks(query_vector: np.ndarray, session_id: str, top_k: int = 20) -> Tuple[List[Dict], List[float]]:
    results = similarity_search(query_vector.tolist(), top_k=top_k, session_id=session_id)
    
    chunks = []
    scores = []
    for match in results.get("matches", []):
        metadata = match["metadata"]
        chunks.append({
            "text": metadata["text"],
            "source": metadata["source"],
            "page": metadata["page"],
            "vector_score": match["score"]
        })
        scores.append(match["score"])
    
    return chunks, scores