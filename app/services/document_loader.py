from langchain_core.document import Document
from langchain_community.document_loaders import PyPDFLoader
from typing import List
from pathlib import Path

def rag_document_loader(file_path:str)->List(Document):
    path=Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"Unsupported file type: {path.suffix}. Only PDF files are supported.")
    try:
        loader=PyPDFLoader(path)
        documents=loader.load()

        for doc in documents:
            doc.metadata['source']=path.name
            return documents
    except Exception as e:
        raise RuntimeError(f"Failed to process pdf file: {path}") from e