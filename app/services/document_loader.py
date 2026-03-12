from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import logging
from pathlib import Path
from typing import Dict

def load_pdf_documents(file_path:str)->List[Document]:
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found at this path: {file_path}")
    if path.suffix.lower()!= ".pdf":
        