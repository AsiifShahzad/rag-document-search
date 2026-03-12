from app.rag.retriever import query_embedding, retrieve_chunks
from app.services.re_ranker import rerank_chunks


def retrieve_context(query: str):
    print(f"\n{'='*60}")
    print(f"[PIPELINE] Starting context retrieval for query: '{query}'")
    print(f"{'='*60}")

    # Step 1: Embed the query
    print(f"\n[PIPELINE] Step 1: Embedding query...")
    query_vector = query_embedding(query)

    # Step 2: Vector search — wide net, top 20 candidates
    print(f"\n[PIPELINE] Step 2: Vector similarity search (top 20)...")
    candidates, vector_scores = retrieve_chunks(query_vector, top_k=20)
    print(f"[PIPELINE] Retrieved {len(candidates)} candidates from vector search")
    
    if len(candidates) == 0:
        print(f"[PIPELINE] ⚠️  WARNING: No candidates found for query!")
        print(f"{'='*60}\n")
        return []

    # Step 3: Rerank — narrow to top 5 accurately
    print(f"\n[PIPELINE] Step 3: Reranking top candidates...")
    final_chunks = rerank_chunks(query, candidates, top_k=5)
    
    print(f"\n[PIPELINE] Final ranked results:")
    for i, chunk in enumerate(final_chunks):
        rerank_score = chunk.get('rerank_score', chunk.get('vector_score', 0))
        vector_score = chunk.get('vector_score', 'N/A')
        fallback = "⚠️ (FALLBACK)" if chunk.get('rerank_fallback', False) else ""
        print(f"  [{i+1}] Final Score: {rerank_score:.4f} | Vector: {vector_score} {fallback}")
        print(f"      Source: {chunk.get('source', 'unknown')} | Page: {chunk.get('page', 'unknown')}")
        print(f"      Text: {chunk.get('text', '')[:80]}...")

    print(f"\n[PIPELINE] Context retrieval complete. Returning {len(final_chunks)} chunks.")
    print(f"[PIPELINE] Note: If FALLBACK shown above, reranker scores were too low,")
    print(f"[PIPELINE]       using vector search scores instead as fallback.")
    print(f"{'='*60}\n")

    return final_chunks