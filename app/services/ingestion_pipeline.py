from app.services.document_loader import rag_document_loader
from app.services.embeddings import document_embedding
from app.services.chunking import split_documents
from app.services.vector_store import insert_embeddings

from typing import List, Tuple
from langchain_core.documents import Document


def data_ingestion(file_path: str, session_id: str) -> Tuple[List[Document], List[List[float]]]:
    documents = rag_document_loader(file_path)
    chunks = split_documents(documents)
    embeddings = document_embedding(chunks)

    metadata = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source"),
            "page": doc.metadata.get("page")
        }
        for doc in chunks
    ]

    insert_embeddings(embeddings, metadata, session_id)
    return chunks, embeddings