from typing import List


def build_prompt(chunks: List[dict], question: str) -> str:

    context_parts = []

    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page")

        citation = f"(Source: {source}, Page: {page})"

        context_parts.append(f"{text}\n{citation}")

    context = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{question}

Answer with sources.
"""

    return prompt