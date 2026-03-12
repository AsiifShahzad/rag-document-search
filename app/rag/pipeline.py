from app.rag.retriever import query_embedding, retrieve_chunks
from app.services.re_ranker import rerank_chunks


def retrieve_context(query: str):
    query_vector = query_embedding(query)
    candidates, vector_scores = retrieve_chunks(query_vector, top_k=20)
    
    if len(candidates) == 0:
        return []

    final_chunks = rerank_chunks(query, candidates, top_k=5)
    return final_chunks