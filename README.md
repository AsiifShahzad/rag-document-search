# RAG Document Search Engine

An AI-powered document Q&A backend that lets users upload PDFs and have accurate, context-aware conversations about their content.

Live Demo: https://document-chat-frontend-kappa.vercel.app/

---

## Problem

Extracting information from documents requires either reading them in full or relying on keyword search that misses context and meaning. Neither approach scales when documents are long, technical, or numerous.

---

## Impact

- Users ask natural language questions and get accurate answers grounded strictly in the uploaded document
- Multi-stage retrieval (coarse search followed by neural re-ranking) improves answer precision over basic vector search
- Session-based architecture supports multiple users with isolated document contexts
- Reduces information retrieval time significantly compared to manual reading

---

## Solution

A RAG pipeline built on FastAPI that processes PDFs end-to-end — from ingestion to grounded answer generation.

Uploaded PDFs are split into semantic chunks and embedded using BAAI/bge-small-en (384 dimensions). Embeddings are stored in Pinecone for similarity search. At query time, the top-k chunks are retrieved and re-ranked using a neural re-ranker for precision, then passed as context to a Groq-hosted LLM (Llama 3.3-70B primary, with Llama 3.1-8B and Gemma2-9B as fallbacks) which generates answers grounded only in the document content.

---

## Tech Stack

FastAPI · Groq (Llama 3.3-70B) · BAAI/bge-small-en · Pinecone · Sentence-Transformers · LangChain · PyPDF · Docker

Deployed on Docker with CORS configured for Vercel frontend integration.

---

## Author

**Asif Shahzad** — AI/ML Engineer  
[Portfolio](https://asiifshahzad.vercel.app) · [LinkedIn](https://www.linkedin.com/in/asiifshahzad) · [Email](mailto:shahzadasif041@gmail.com)
