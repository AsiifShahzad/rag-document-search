from app.services.document_loader import rag_document_loader
from app.services.embeddings import document_embedding
from app.services.chunking import split_documents
from app.services.vector_store import insert_embeddings

from typing import List, Tuple
from langchain_core.documents import Document


def data_ingestion(file_path: str) -> Tuple[List[Document], List[List[float]]]:

    print(f"\n{'='*60}")
    print(f"[INGESTION] Starting document ingestion")
    print(f"[INGESTION] File: {file_path}")
    print(f"{'='*60}")

    documents = rag_document_loader(file_path)
    print(f"\n[INGESTION] ✓ Loaded {len(documents)} pages from PDF")
    for i, doc in enumerate(documents[:3]):
        print(f"  Page {i}: {len(doc.page_content)} characters")

    chunks = split_documents(documents)
    print(f"\n[INGESTION] ✓ Split into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:5]):
        print(f"  Chunk {i+1}: {len(chunk.page_content)} chars | Page: {chunk.metadata.get('page', 'unknown')}")
    if len(chunks) > 5:
        print(f"  ... and {len(chunks) - 5} more chunks")

    embeddings = document_embedding(chunks)
    print(f"\n[INGESTION] ✓ Generated {len(embeddings)} embeddings")
    if embeddings:
        import numpy as np
        embedding_array = np.array(embeddings[0])
        print(f"  Embedding dimensions: {embedding_array.shape}")
        print(f"  Sample embedding: {embeddings[0][:5]}")
        print(f"  Embedding norm: {np.linalg.norm(embedding_array):.4f}")

    metadata = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in chunks
    ]

    print(f"\n[INGESTION] Storing vectors in Pinecone...")
    insert_embeddings(embeddings, metadata)
    print(f"[INGESTION] ✓ Vectors stored successfully")
    
    print(f"\n[INGESTION] Document ingestion COMPLETE")
    print(f"{'='*60}\n")

    return chunks, embeddings