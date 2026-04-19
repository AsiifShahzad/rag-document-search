"""
Application configuration settings
"""
import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

# API Configuration
class Settings:
    # App info
    APP_NAME = "DocumentChat RAG Backend"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 7860))
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "https://document-chat-frontend-kappa.vercel.app",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS = ["*"]
    CORS_MAX_AGE = 600
    
    # File upload
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR = "data/documents"
    
    # External APIs
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX", "document-embeddings")
    
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    
    @classmethod
    def is_production(cls) -> bool:
        return not cls.DEBUG
    
    @classmethod
    def get_log_level(cls) -> str:
        return "DEBUG" if cls.DEBUG else "INFO"


settings = Settings()
