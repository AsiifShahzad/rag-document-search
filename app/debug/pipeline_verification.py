"""
Standalone debug utility for RAG pipeline verification
Run this script to test and verify each phase of the RAG pipeline
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.embeddings import get_embedding_model
from app.services.vector_store import similarity_search, index as pinecone_index, INDEX_NAME
from app.services.chunking import split_documents
from app.services.re_ranker import get_reranker
from app.rag.pipeline import retrieve_context


class DebugReport:
    def __init__(self):
        self.phases = {}
        self.issues = []
    
    def add_phase(self, phase_name: str, data: dict):
        self.phases[phase_name] = data
    
    def add_issue(self, severity: str, phase: str, message: str):
        self.issues.append({
            "severity": severity,
            "phase": phase,
            "message": message
        })
    
    def print_report(self):
        print("\n" + "="*80)
        print("RAG PIPELINE DEBUG REPORT")
        print("="*80)
        
        for phase_name, data in self.phases.items():
            print(f"\n{'='*80}")
            print(f"PHASE: {phase_name}")
            print(f"{'='*80}")
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
        
        if self.issues:
            print(f"\n{'='*80}")
            print("IDENTIFIED ISSUES")
            print(f"{'='*80}")
            critical = [i for i in self.issues if i["severity"] == "CRITICAL"]
            warning = [i for i in self.issues if i["severity"] == "WARNING"]
            info = [i for i in self.issues if i["severity"] == "INFO"]
            
            if critical:
                print("\n🔴 CRITICAL ISSUES:")
                for issue in critical:
                    print(f"  [{issue['phase']}] {issue['message']}")
            
            if warning:
                print("\n🟡 WARNINGS:")
                for issue in warning:
                    print(f"  [{issue['phase']}] {issue['message']}")
            
            if info:
                print("\n🔵 INFO:")
                for issue in info:
                    print(f"  [{issue['phase']}] {issue['message']}")
        else:
            print(f"\n{'='*80}")
            print("✅ NO ISSUES FOUND - Pipeline appears healthy!")
            print(f"{'='*80}")


def phase1_embedding_model_verification():
    """Phase 1: Verify embedding model works"""
    print("\n" + "="*80)
    print("PHASE 1: EMBEDDING MODEL VERIFICATION")
    print("="*80)
    
    report = DebugReport()
    
    try:
        print("\nTesting embedding model initialization...")
        model = get_embedding_model()
        print("✓ Embedding model loaded successfully")
        
        print("\nGenerating test embedding...")
        test_query = "What are masked autoencoders"
        test_embedding = model.embed_query(test_query)
        test_array = np.array(test_embedding)
        
        print(f"✓ Test embedding generated")
        print(f"  Query: '{test_query}'")
        print(f"  Embedding dimensions: {test_array.shape}")
        print(f"  Embedding norm: {np.linalg.norm(test_array):.6f}")
        print(f"  Sample values (first 10): {test_array[:10]}")
        print(f"  Min value: {test_array.min():.6f}")
        print(f"  Max value: {test_array.max():.6f}")
        print(f"  Mean value: {test_array.mean():.6f}")
        
        # Check for common issues
        if test_array.shape[0] == 0:
            report.add_issue("CRITICAL", "Embedding Model", "Embedding is empty")
        
        if np.all(test_array == 0):
            report.add_issue("CRITICAL", "Embedding Model", "Embedding is all zeros")
        
        if np.any(np.isnan(test_array)):
            report.add_issue("CRITICAL", "Embedding Model", "Embedding contains NaN values")
        
        report.add_phase("Embedding Model", {
            "status": "✓ Working",
            "dimensions": int(test_array.shape[0]),
            "embedding_norm": float(np.linalg.norm(test_array)),
            "has_zeros": bool(np.all(test_array == 0)),
            "has_nans": bool(np.any(np.isnan(test_array)))
        })
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        report.add_issue("CRITICAL", "Embedding Model", f"Failed to load embedding model: {str(e)}")
        report.add_phase("Embedding Model", {"status": "✗ Failed", "error": str(e)})
    
    return report


def phase2_pinecone_connection_verification():
    """Phase 2: Verify Pinecone connectivity and existing vectors"""
    print("\n" + "="*80)
    print("PHASE 2: PINECONE CONNECTION & DATA VERIFICATION")
    print("="*80)
    
    report = DebugReport()
    
    try:
        print("\nTesting Pinecone connection...")
        dummy_vector = [0.0] * 384
        result = similarity_search(dummy_vector, top_k=1)
        print("✓ Pinecone connection successful")
        
        print("\nQuerying index statistics...")
        stats = pinecone_index.describe_index_stats()
        total_vectors = stats.total_vector_count
        
        print(f"✓ Index stats retrieved")
        print(f"  Total vectors in index: {total_vectors}")
        
        if total_vectors == 0:
            report.add_issue("CRITICAL", "Pinecone", "No vectors found in index. Documents may not be ingested.")
        else:
            print(f"\nRetrieving sample vectors...")
            sample_results = similarity_search(dummy_vector, top_k=min(5, total_vectors))
            
            print(f"✓ Retrieved {len(sample_results.get('matches', []))} sample vectors")
            for i, match in enumerate(sample_results.get("matches", [])):
                metadata = match.get("metadata", {})
                print(f"\n  Sample {i+1}:")
                print(f"    ID: {match.get('id', 'N/A')}")
                print(f"    Score: {match.get('score', 'N/A'):.4f}")
                print(f"    Source: {metadata.get('source', 'N/A')}")
                print(f"    Page: {metadata.get('page', 'N/A')}")
                print(f"    Text preview: {metadata.get('text', '')[:80]}...")
        
        report.add_phase("Pinecone Connection", {
            "status": "✓ Connected",
            "total_vectors": int(total_vectors),
            "index_name": INDEX_NAME,
            "dimension": 384
        })
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        report.add_issue("CRITICAL", "Pinecone", f"Failed to connect to Pinecone: {str(e)}")
        report.add_phase("Pinecone Connection", {"status": "✗ Failed", "error": str(e)})
    
    return report


def phase3_retrieval_test(test_queries: List[str] = None):
    """Phase 3: Test vector similarity search and retrieval"""
    print("\n" + "="*80)
    print("PHASE 3: RETRIEVAL & SIMILARITY SEARCH VERIFICATION")
    print("="*80)
    
    if test_queries is None:
        test_queries = [
            "What are masked autoencoders",
            "Vision learning CNN encoders",
            "autoencoder training",
            "He Masked paper"
        ]
    
    report = DebugReport()
    retrieval_results = {}
    
    try:
        model = get_embedding_model()
        
        for query in test_queries:
            print(f"\n{'─'*60}")
            print(f"Testing query: '{query}'")
            print(f"{'─'*60}")
            
            # Generate embedding
            query_vector = model.embed_query(query)
            query_array = np.array(query_vector)
            
            # Search
            results = similarity_search(query_array.tolist(), top_k=10)
            matches = results.get("matches", [])
            
            print(f"Retrieved {len(matches)} results")
            
            scores = []
            for i, match in enumerate(matches):
                score = match.get("score", 0)
                text = match.get("metadata", {}).get("text", "")
                scores.append(score)
                print(f"  [{i+1}] Score: {score:.4f} | Text: {text[:80]}...")
            
            retrieval_results[query] = {
                "count": len(matches),
                "scores": scores,
                "avg_score": float(np.mean(scores)) if scores else 0,
                "max_score": float(np.max(scores)) if scores else 0,
                "min_score": float(np.min(scores)) if scores else 0
            }
            
            # Check for issues
            if len(matches) == 0:
                report.add_issue("WARNING", "Retrieval", f"Query '{query}' returned no results")
            elif max(scores) < 0.3:
                report.add_issue("WARNING", "Retrieval", f"Query '{query}' returned low similarity scores (max: {max(scores):.4f})")
        
        report.add_phase("Retrieval Performance", retrieval_results)
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        report.add_issue("CRITICAL", "Retrieval", f"Retrieval test failed: {str(e)}")
        report.add_phase("Retrieval Performance", {"status": "✗ Failed", "error": str(e)})
    
    return report


def phase4_reranking_test():
    """Phase 4: Test reranking functionality"""
    print("\n" + "="*80)
    print("PHASE 4: RERANKING VERIFICATION")
    print("="*80)
    
    report = DebugReport()
    
    try:
        print("\nLoading reranker...")
        reranker = get_reranker()
        print("✓ Reranker loaded")
        
        print("\nTesting reranker with sample pairs...")
        test_pairs = [
            ("What are masked autoencoders", "Masked autoencoders are trained by masking parts"),
            ("Vision learning", "This approach works for computer vision tasks"),
            ("What is weather", "The sky is blue today")
        ]
        
        scores = reranker.predict(test_pairs)
        
        for (query, text), score in zip(test_pairs, scores):
            print(f"  Score: {score:.4f} | Query: '{query}' | Text: '{text}'")
        
        report.add_phase("Reranking", {
            "status": "✓ Working",
            "model": "BAAI/bge-reranker-base",
            "test_scores": [float(s) for s in scores]
        })
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        report.add_issue("WARNING", "Reranking", f"Reranker test failed: {str(e)}")
        report.add_phase("Reranking", {"status": "✗ Failed", "error": str(e)})
    
    return report


def phase5_end_to_end_test(test_queries: List[str] = None):
    """Phase 5: End-to-end pipeline test"""
    print("\n" + "="*80)
    print("PHASE 5: END-TO-END PIPELINE TEST")
    print("="*80)
    
    if test_queries is None:
        test_queries = [
            "What are masked autoencoders",
            "How does this approach work",
            "What is the title of this paper"
        ]
    
    report = DebugReport()
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        print(f"Testing query: '{query}'")
        print(f"{'─'*60}")
        
        try:
            chunks = retrieve_context(query)
            
            if not chunks:
                print("✗ No chunks retrieved")
                report.add_issue("WARNING", "End-to-End", f"No chunks retrieved for query: '{query}'")
            else:
                print(f"✓ Retrieved {len(chunks)} chunks")
                for i, chunk in enumerate(chunks[:3]):
                    rerank_score = chunk.get('rerank_score', 'N/A')
                    vector_score = chunk.get('vector_score', 'N/A')
                    print(f"  [{i+1}] Rerank: {rerank_score} | Vector: {vector_score} | Page: {chunk.get('page')}")
                    print(f"       Text: {chunk.get('text', '')[:100]}...")
        
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            report.add_issue("CRITICAL", "End-to-End", f"Pipeline failed for query '{query}': {str(e)}")
    
    return report


def main():
    """Run all verification phases"""
    print("\n" + "="*80)
    print("RAG PIPELINE COMPREHENSIVE DEBUG VERIFICATION")
    print("="*80)
    
    all_reports = []
    
    # Phase 1: Embedding Model
    report1 = phase1_embedding_model_verification()
    all_reports.append(report1)
    
    # Phase 2: Pinecone Connection
    report2 = phase2_pinecone_connection_verification()
    all_reports.append(report2)
    
    # Phase 3: Retrieval
    report3 = phase3_retrieval_test()
    all_reports.append(report3)
    
    # Phase 4: Reranking
    report4 = phase4_reranking_test()
    all_reports.append(report4)
    
    # Phase 5: End-to-end
    report5 = phase5_end_to_end_test()
    all_reports.append(report5)
    
    # Combine all reports
    combined_report = DebugReport()
    for report in all_reports:
        combined_report.phases.update(report.phases)
        combined_report.issues.extend(report.issues)
    
    combined_report.print_report()
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    critical_count = len([i for i in combined_report.issues if i["severity"] == "CRITICAL"])
    warning_count = len([i for i in combined_report.issues if i["severity"] == "WARNING"])
    
    print(f"Critical Issues: {critical_count}")
    print(f"Warnings: {warning_count}")
    
    if critical_count == 0 and warning_count == 0:
        print("\n✅ All checks passed! Pipeline is healthy.")
    else:
        print("\n⚠️  Issues detected. Review above for details.")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
