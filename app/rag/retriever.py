from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def query_embedding(query: str) -> np.ndarray:
    model = get_embedding_model()
    vector = model.embed_query(query)
    vector_array = np.array(vector)
    return vector_array


def retrieve_chunks(query_vector: np.ndarray, top_k: int = 20) -> Tuple[List[Dict], List[float]]:
    results = similarity_search(query_vector.tolist(), top_k=top_k)
    
    chunks = []
    scores = []
    for match in results.get("matches", []):
        metadata = match["metadata"]
        chunk = {
            "text": metadata["text"],
            "source": metadata["source"],
            "page": metadata["page"],
            "vector_score": match["score"]
        }
        chunks.append(chunk)
        scores.append(match["score"])
    
    return chunks, scores