from typing import List


def build_prompt(chunks: List[dict], question: str) -> str:

    print(f"\n{'='*60}")
    print(f"[PROMPT_BUILDER] Building prompt with {len(chunks)} chunks")
    print(f"[PROMPT_BUILDER] Question: '{question}'")
    print(f"{'='*60}")

    context_parts = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        rerank_fallback = chunk.get("rerank_fallback", False)
        
        context_parts.append(f"[Chunk {i+1} | Source: {source}, Page: {page}]\n{text}")
        
        print(f"\n[PROMPT_BUILDER] Chunk {i+1}:")
        print(f"  Source: {source}")
        print(f"  Page: {page}")
        print(f"  Length: {len(text)} characters")
        if rerank_fallback:
            print(f"  Note: Using fallback ranking (vector search)")
        print(f"  Preview: {text[:100]}...")

    context = "\n\n".join(context_parts)
    print(f"\n[PROMPT_BUILDER] Total context length: {len(context)} characters")

    # Use a simple, direct prompt format that LLMs understand well
    prompt = f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""

    print(f"\n[PROMPT_BUILDER] Final prompt size: {len(prompt)} characters")
    print(f"[PROMPT_BUILDER] ✓ Prompt ready for LLM")
    print(f"{'='*60}\n")

    return prompt