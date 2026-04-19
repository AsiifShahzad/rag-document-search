import logging
import sys
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.api.app import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocumentChat RAG Backend",
    description="RAG-based document search and Q&A system"
)

# 1. Trust proxy headers (important for Render)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# 2. CORS origins — uncomment specific origins for production
allowed_origins = [
    "https://document-chat-frontend-kappa.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",   # ← ADD THIS (Vite dev server)
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",   # ← ADD THIS too
    "http://127.0.0.1:8000",
]
# Uncomment below line (and comment above) only for quick testing:
# allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,        # ← was using undefined variable
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=600,
    expose_headers=["Content-Type", "X-Process-Time"],
)

logger.info(f"CORS configured for origins: {allowed_origins}")

# Include router AFTER middleware setup
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Backend started successfully")
    logger.info(f"CORS allowed origins: {allowed_origins}")

    logger.info("Preloading ML models...")
    try:
        from app.services.embeddings import get_embedding_model
        from app.services.re_ranker import get_reranker
        from app.services.vector_store import similarity_search

        logger.info("Loading embedding model...")
        model = get_embedding_model()
        logger.info("✓ Embedding model loaded")

        logger.info("Loading reranker model...")
        reranker = get_reranker()
        logger.info("✓ Reranker model loaded")

        logger.info("Testing vector store connection...")
        dummy_vector = [0.0] * 384
        similarity_search(dummy_vector, top_k=1)
        logger.info("✓ Vector store connected")

        logger.info("✓ All models loaded successfully! Backend ready for requests.")
    except Exception as e:
        logger.error(f"Failed to preload models: {str(e)}", exc_info=True)
        logger.warning("Backend will run in degraded mode until models load on first request")

@app.get("/")
async def root():
    return {"status": "ok", "message": "DocumentChat RAG Backend is running"}