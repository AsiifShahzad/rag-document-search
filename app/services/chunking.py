from typing import List
from langchain_core.document import Document
from langchain_core.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def split_documents(documents: List[Document])->List[Document]:
    if not documents:
        raise ValueError("No documents to split.")
    chunks = text_splitter.split_documents(documents)
    return chunks
    