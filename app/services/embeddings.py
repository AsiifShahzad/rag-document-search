from typing import List
from langchain_core.document import Document
from langchain_core.embeddings import HuggingFaceEmbeddings

#global reuseable embedding model object
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        test_vector = _embedding_model.embed_query("test")
        if not test_vector or len(test_vector) == 0: 
            raise RuntimeError("Failed to initialize embedding model: no output from test embedding.")
    return _embedding_model

def document_embedding(documents: List[Document])->List[List[float]]:
    if not documents:
        raise ValueError("No documents to embed.")
    embedding_model = get_embedding_model()

    texts= [doc.page_content for doc in documents]
    embeddings=embedding_model.embed_documents(texts)
    return embeddings

