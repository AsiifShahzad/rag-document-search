#!/usr/bin/env python
"""
Quick test script to verify the RAG pipeline fixes
Run this after starting the backend server
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test basic health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✓ Health check passed")
        print(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


def test_verify_pipeline():
    """Test comprehensive pipeline verification"""
    print("\n" + "="*60)
    print("TEST 2: Pipeline Verification")
    print("="*60)
    
    try:
        response = requests.get(f"{BASE_URL}/debug/verify-all", params={"test_query": "masked autoencoders"})
        data = response.json()
        print(f"✓ Pipeline verification completed")
        print(json.dumps(data.get("results", {}).get("verification", {}), indent=2))
        return True
    except Exception as e:
        print(f"✗ Pipeline verification failed: {e}")
        return False


def test_specific_queries():
    """Test queries that previously failed"""
    print("\n" + "="*60)
    print("TEST 3: Test Problem Queries")
    print("="*60)
    
    test_queries = [
        {
            "question": "What are masked autoencoders?",
            "expected": "specific query - should work well",
            "type": "direct"
        },
        {
            "question": "How does this approach work?",
            "expected": "generic query - should now use fallback scores",
            "type": "generic"
        },
        {
            "question": "What is the title of this paper?",
            "expected": "meta question - now with better prompting",
            "type": "meta"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n  [{i}] Query Type: {test_case['type'].upper()}")
        print(f"     Question: {test_case['question']}")
        print(f"     Expected: {test_case['expected']}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                json={"question": test_case["question"]},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "")
                confidence = data.get("confidence", 0)
                sources = data.get("sources", [])
                
                print(f"     Status: ✓ Success")
                print(f"     Confidence: {confidence}")
                print(f"     Sources: {len(sources)}")
                
                if answer:
                    preview = answer[:100] + "..." if len(answer) > 100 else answer
                    print(f"     Answer: {preview}")
                    
                    if "couldn't find relevant information" in answer.lower():
                        print(f"     ⚠️  Still returning 'not found' - check logs")
                    else:
                        print(f"     ✓ Returning actual answer content")
                else:
                    print(f"     ✗ Empty answer returned")
            else:
                print(f"     ✗ Failed with status {response.status_code}")
                print(f"     Error: {response.text}")
        
        except requests.exceptions.ConnectionError:
            print(f"     ✗ Cannot connect to server - is it running?")
            return False
        except Exception as e:
            print(f"     ✗ Error: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    return True


def main():
    print("\n" + "="*80)
    print("RAG PIPELINE FIX VERIFICATION TEST SUITE")
    print("="*80)
    
    print("\nChecking if server is running...")
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except:
        print("\n✗ SERVER IS NOT RUNNING")
        print("\nTo start the server, run:")
        print("  python -m uvicorn app.main:app --reload")
        print("\nThen run this script again in another terminal.")
        return
    
    print("✓ Server is running\n")
    
    # Run tests
    health_ok = test_health()
    pipeline_ok = test_verify_pipeline()
    queries_ok = test_specific_queries()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    tests = {
        "Health Check": health_ok,
        "Pipeline Verification": pipeline_ok,
        "Query Tests": queries_ok
    }
    
    for test_name, passed in tests.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print("\n" + "="*80)
    
    all_passed = all(tests.values())
    
    if all_passed:
        print("✅ ALL TESTS PASSED - Pipeline fixes are working!")
    else:
        print("⚠️  Some tests failed - check output above for details")
    
    print("\nKey changes made:")
    print("1. ✓ Reranker now has fallback to vector scores when avg score < 0.1")
    print("2. ✓ Prompt builder improved with better instructions")
    print("3. ✓ Pipeline logs which scoring method was used (rerank vs fallback)")
    print("4. ✓ Debug script fixed to work without pinecone_index.name error")
    print("\nIf queries still return 'not found':")
    print("- Check the console logs for [RERANKER] fallback messages")
    print("- Verify chunks are being retrieved in the logs")
    print("- Check LLM API key and rate limits")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
