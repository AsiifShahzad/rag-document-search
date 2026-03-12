from document_loader import rag_document_loader
from embeddings import document_embedding
from chunking import split_documents
from vector_store import upsert_embeddings

from typing import List, Tuple
from langchain_core.documents import Document
from pathlib import Path


def data_ingestion(file_path: str) -> Tuple[List[Document], List[List[float]]]:

    print(f"Starting data ingestion for file: {file_path}")

    documents = rag_document_loader(file_path)
    print(f"Loaded {len(documents)} documents")

    print("Splitting documents into chunks")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Generating embeddings")
    embeddings = document_embedding(chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks")

    metadata = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in chunks
    ]

    print("Uploading embeddings to Pinecone")
    upsert_embeddings(embeddings, metadata)

    print("Ingestion completed")

    return chunks, embeddings


if __name__ == "__main__":

    example_pdf = Path("data/documents/example.pdf")

    if not example_pdf.exists():
        print(f"File does not exist: {example_pdf}")

    else:

        chunks, embeddings = data_ingestion(str(example_pdf))

        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i+1} metadata: {chunk.metadata}")
            print(f"Chunk text preview: {chunk.page_content[:100]}...")

        print(f"\nTotal embeddings generated: {len(embeddings)}")