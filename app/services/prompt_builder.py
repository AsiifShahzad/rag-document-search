from typing import List


def build_prompt(chunks: List[dict], question: str) -> str:

    context_parts = []

    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        context_parts.append(f"[Chunk {i+1} | Source: {source}, Page: {page}]\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""Use the following document excerpts to answer the question at the end.
Do NOT say "what is your question" or ask for clarification.
Do NOT introduce yourself.
Just directly answer the question using the context below.
If the answer is not in the context, say "I couldn't find relevant information in the document."

DOCUMENT EXCERPTS:
{context}

QUESTION: {question}

DIRECT ANSWER:"""

    return prompt