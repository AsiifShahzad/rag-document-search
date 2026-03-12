from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

def query_embedding(query: str) -> np.ndarray:
    print(f"\n{'='*60}")
    print(f"[RETRIEVER] Generating embedding for query: '{query}'")
    model = get_embedding_model()
    vector = model.embed_query(query)
    vector_array = np.array(vector)
    print(f"[RETRIEVER] Embedding generated - dimensions: {vector_array.shape}")
    print(f"[RETRIEVER] Sample values: {vector_array[:5]}")
    print(f"[RETRIEVER] Vector norm: {np.linalg.norm(vector_array):.4f}")
    return vector_array


def retrieve_chunks(query_vector: np.ndarray, top_k: int = 20) -> Tuple[List[Dict], List[float]]:
    print(f"\n[RETRIEVER] Searching Pinecone for top {top_k} similar chunks...")
    print(f"[RETRIEVER] Query vector norm: {np.linalg.norm(query_vector):.4f}")
    
    results = similarity_search(query_vector.tolist(), top_k=top_k)
    
    print(f"[RETRIEVER] Retrieved {len(results.get('matches', []))} results from Pinecone")
    
    chunks = []
    scores = []
    for i, match in enumerate(results.get("matches", [])):
        metadata = match["metadata"]
        chunk = {
            "text": metadata["text"],
            "source": metadata["source"],
            "page": metadata["page"],
            "vector_score": match["score"]
        }
        chunks.append(chunk)
        scores.append(match["score"])
        print(f"  [{i+1}] Score: {match['score']:.4f} | Source: {metadata['source']} | Page: {metadata['page']}")
        print(f"      Text preview: {metadata['text'][:80]}...")
    
    print(f"[RETRIEVER] Min score: {min(scores) if scores else 'N/A':.4f}")
    print(f"[RETRIEVER] Max score: {max(scores) if scores else 'N/A':.4f}")
    print(f"[RETRIEVER] Avg score: {sum(scores)/len(scores) if scores else 'N/A':.4f}")
    
    return chunks, scores