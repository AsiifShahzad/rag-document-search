from typing import List
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import hashlib
load_dotenv() 

INDEX_NAME = "bge-small-index"
DIMENSION = 384

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def generate_unique_id(session_id: str, document_source: str, chunk_index: int) -> str:
    """Generate a unique ID based on session, document source and chunk index"""
    hash_input = f"{session_id}_{document_source}_{chunk_index}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"{session_id}_{document_source.replace('.pdf', '')}_{chunk_index}_{hash_value}"

def insert_embeddings(embeddings: List[List[float]], metadata: List[dict], session_id: str):
    """Insert embeddings into Pinecone with unique IDs tied to session and document"""
    if not embeddings or not metadata:
        raise ValueError("Embeddings and metadata cannot be empty")
    
    if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
        raise ValueError("Valid session_id is required")
    
    try:
        vectors = []
        for i, (embedding, meta) in enumerate(zip(embeddings, metadata)):
            document_source = meta.get("source", "unknown")
            if not document_source or not isinstance(document_source, str):
                raise ValueError(f"Invalid document source at index {i}")
            
            unique_id = generate_unique_id(session_id, document_source, i)
            
            vectors.append(
                {
                    "id": unique_id,
                    "values": embedding,
                    "metadata": {
                        **meta,
                        "session_id": session_id,
                        "document_id": unique_id,
                        "chunk_index": i
                    }
                }
            )
        
        # Upsert in batches for large datasets
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch)
        
        return True
    except Exception as e:
        raise RuntimeError(f"Failed to insert embeddings into Pinecone: {str(e)}") from e

def delete_document_embeddings(document_source: str) -> dict:
    """Delete all embeddings for a specific document from Pinecone"""
    try:
        # Query to find all vectors with this document source
        results = index.query(
            vector=[0.0] * DIMENSION,  # Dummy vector
            top_k=10000,
            include_metadata=True,
            filter={"source": {"$eq": document_source}}
        )
        
        # Delete all matching vectors
        ids_to_delete = [match["id"] for match in results.get("matches", [])]
        
        if ids_to_delete:
            index.delete(ids=ids_to_delete)
            return {
                "success": True,
                "document": document_source,
                "deleted_count": len(ids_to_delete)
            }
        else:
            return {
                "success": True,
                "document": document_source,
                "deleted_count": 0,
                "message": "No embeddings found for this document"
            }
    except Exception as e:
        return {
            "success": False,
            "document": document_source,
            "error": str(e)
        }

def delete_session_embeddings(session_id: str) -> dict:
    """Delete all embeddings for a specific session from Pinecone"""
    if not session_id or not isinstance(session_id, str) or len(session_id.strip()) == 0:
        return {
            "success": False,
            "session_id": session_id,
            "error": "Invalid session_id"
        }
    
    try:
        total_deleted = 0
        
        # Use pagination to handle sessions with many embeddings
        # Query in batches to get all embeddings for this session
        for batch_num in range(100):  # Max 100 batches = 1M embeddings
            results = index.query(
                vector=[0.0] * DIMENSION,  # Dummy vector
                top_k=10000,
                include_metadata=True,
                filter={"session_id": {"$eq": session_id}}
            )
            
            ids_to_delete = [match["id"] for match in results.get("matches", [])]
            
            if not ids_to_delete:
                break  # No more embeddings for this session
            
            index.delete(ids=ids_to_delete)
            total_deleted += len(ids_to_delete)
        
        return {
            "success": True,
            "session_id": session_id,
            "deleted_count": total_deleted
        }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "error": f"Database error: {str(e)}"
        }

def similarity_search(query_embedding: List[float], top_k: int = 20, session_id: str = None) -> dict:
    filter_dict = None
    if session_id:
        filter_dict = {"session_id": {"$eq": session_id}}
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict
    )
    return results
