from typing import List


def build_prompt(chunks: List[dict], question: str) -> str:
    context_parts = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        
        context_parts.append(f"[Chunk {i+1} | Source: {source}, Page: {page}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""

    return prompt