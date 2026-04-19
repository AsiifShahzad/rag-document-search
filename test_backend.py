#!/usr/bin/env python3
"""
CORS & Backend Verification Script
Run this locally to test all endpoints before deploying to Render
"""

import subprocess
import json
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
FRONTEND_URL = "https://document-chat-frontend-kappa.vercel.app"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_info(text):
    print(f"  {text}")

def check_backend_running():
    """Check if backend is running"""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", f"{BASE_URL}/"],
            capture_output=True,
            timeout=5
        )
        status_code = result.stdout.decode().strip()
        if status_code == "200":
            print_success("Backend is running")
            return True
        else:
            print_error(f"Backend returned status {status_code}")
            return False
    except Exception as e:
        print_error(f"Backend is not running: {e}")
        print_info("Start it with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        return False

def test_health_endpoint():
    """Test /health endpoint"""
    print_header("Testing /health Endpoint")
    
    try:
        result = subprocess.run(
            ["curl", "-s", 
             "-H", f"Origin: {FRONTEND_URL}",
             "-H", "Accept: application/json",
             f"{BASE_URL}/health"],
            capture_output=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print_error(f"Request failed: {result.stderr.decode()}")
            return False
        
        response = json.loads(result.stdout.decode())
        print_info(f"Response: {json.dumps(response, indent=2)}")
        
        if response.get("overall") == "healthy":
            print_success("All components healthy")
            return True
        else:
            print_warning(f"Degraded health: {response}")
            for key, value in response.items():
                if value != "ok":
                    print_warning(f"  {key}: {value}")
            return False
            
    except Exception as e:
        print_error(f"Health check failed: {e}")
        return False

def test_cors_headers():
    """Test CORS headers with OPTIONS request"""
    print_header("Testing CORS Headers")
    
    try:
        # Test OPTIONS preflight
        result = subprocess.run(
            ["curl", "-s", "-X", "OPTIONS",
             "-H", f"Origin: {FRONTEND_URL}",
             "-H", "Access-Control-Request-Method: POST",
             "-H", "Access-Control-Request-Headers: Content-Type",
             "-D", "-",  # Show headers
             f"{BASE_URL}/upload"],
            capture_output=True,
            timeout=30
        )
        
        output = result.stdout.decode()
        headers = output.split('\n')
        
        cors_headers = {
            "allow-origin": None,
            "allow-methods": None,
            "allow-headers": None,
            "allow-credentials": None
        }
        
        for header in headers:
            header_lower = header.lower()
            if "access-control-allow-origin" in header_lower:
                cors_headers["allow-origin"] = header.split(": ")[1].strip() if ": " in header else ""
            elif "access-control-allow-methods" in header_lower:
                cors_headers["allow-methods"] = header.split(": ")[1].strip() if ": " in header else ""
            elif "access-control-allow-headers" in header_lower:
                cors_headers["allow-headers"] = header.split(": ")[1].strip() if ": " in header else ""
            elif "access-control-allow-credentials" in header_lower:
                cors_headers["allow-credentials"] = header.split(": ")[1].strip() if ": " in header else ""
        
        print_info("CORS Headers:")
        for key, value in cors_headers.items():
            if value:
                print_info(f"  {key}: {value}")
                print_success(f"{key} present")
            else:
                print_warning(f"{key} missing")
        
        if cors_headers["allow-origin"] == FRONTEND_URL:
            print_success("Frontend URL correctly allowed")
            return True
        else:
            print_error(f"Frontend URL not allowed. Got: {cors_headers['allow-origin']}")
            return False
            
    except Exception as e:
        print_error(f"CORS test failed: {e}")
        return False

def test_file_upload():
    """Test file upload endpoint"""
    print_header("Testing /upload Endpoint")
    
    # Create a test PDF file
    test_pdf_path = Path("test_sample.pdf")
    if not test_pdf_path.exists():
        print_warning(f"Test PDF not found at {test_pdf_path}")
        print_info("Creating minimal PDF for testing...")
        
        # Create minimal PDF content
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello PDF!) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000214 00000 n
0000000301 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
395
%%EOF"""
        
        with open(test_pdf_path, 'wb') as f:
            f.write(pdf_content)
        print_success(f"Test PDF created at {test_pdf_path}")
    
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        result = subprocess.run(
            ["curl", "-s",
             "-F", f"file=@{test_pdf_path}",
             "-F", f"session_id={session_id}",
             "-H", f"Origin: {FRONTEND_URL}",
             f"{BASE_URL}/upload"],
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print_error(f"Upload request failed: {result.stderr.decode()}")
            return False
        
        response = json.loads(result.stdout.decode())
        print_info(f"Response: {json.dumps(response, indent=2)}")
        
        if response.get("success"):
            print_success("File uploaded successfully")
            print_info(f"Session ID: {response['session_id']}")
            print_info(f"Chunks processed: {response['chunks_processed']}")
            return True, response['session_id']
        else:
            print_error(f"Upload failed: {response}")
            return False, None
            
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON response: {e}")
        print_info(f"Response was: {result.stdout.decode()[:200]}")
        return False, None
    except Exception as e:
        print_error(f"Upload test failed: {e}")
        return False, None

def test_ask_endpoint(session_id):
    """Test /ask endpoint"""
    print_header("Testing /ask Endpoint")
    
    if not session_id:
        print_warning("No session ID available, skipping ask test")
        return False
    
    try:
        payload = {
            "question": "What is this document about?",
            "session_id": session_id
        }
        
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "-H", "Content-Type: application/json",
             "-H", f"Origin: {FRONTEND_URL}",
             "-d", json.dumps(payload),
             f"{BASE_URL}/ask"],
            capture_output=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print_error(f"Ask request failed: {result.stderr.decode()}")
            return False
        
        response = json.loads(result.stdout.decode())
        print_info(f"Response: {json.dumps(response, indent=2)}")
        
        if "answer" in response:
            print_success("Got answer")
            print_info(f"Answer: {response['answer'][:100]}...")
            print_info(f"Sources: {len(response.get('sources', []))} found")
            return True
        else:
            print_error(f"No answer in response: {response}")
            return False
            
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON response: {e}")
        print_info(f"Response was: {result.stdout.decode()[:200]}")
        return False
    except Exception as e:
        print_error(f"Ask test failed: {e}")
        return False

def test_cleanup_endpoint(session_id):
    """Test /cleanup endpoint"""
    print_header("Testing /cleanup Endpoint")
    
    if not session_id:
        print_warning("No session ID available, skipping cleanup test")
        return False
    
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST",
             "-F", f"session_id={session_id}",
             "-H", f"Origin: {FRONTEND_URL}",
             f"{BASE_URL}/cleanup"],
            capture_output=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print_error(f"Cleanup request failed: {result.stderr.decode()}")
            return False
        
        response = json.loads(result.stdout.decode())
        print_info(f"Response: {json.dumps(response, indent=2)}")
        
        if response.get("success"):
            print_success("Session cleaned up")
            print_info(f"Deleted count: {response.get('deleted_count', 0)}")
            return True
        else:
            print_error(f"Cleanup failed: {response}")
            return False
            
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON response: {e}")
        return False
    except Exception as e:
        print_error(f"Cleanup test failed: {e}")
        return False

def main():
    print(f"{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║      CORS & Backend Verification Script                   ║")
    print("║      Testing: http://localhost:8000                       ║")
    print("║      Frontend: https://document-chat-frontend-...         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    # Check backend is running
    if not check_backend_running():
        sys.exit(1)
    
    time.sleep(1)  # Give server time to respond
    
    # Run tests
    results = []
    
    results.append(("Health Endpoint", test_health_endpoint()))
    results.append(("CORS Headers", test_cors_headers()))
    
    upload_result, session_id = test_file_upload()
    results.append(("File Upload", upload_result))
    
    if session_id:
        results.append(("Ask Endpoint", test_ask_endpoint(session_id)))
        results.append(("Cleanup Endpoint", test_cleanup_endpoint(session_id)))
    
    # Summary
    print_header("Test Summary")
    passed = 0
    failed = 0
    
    for test_name, result in results:
        if result:
            print_success(test_name)
            passed += 1
        else:
            print_error(test_name)
            failed += 1
    
    print_info(f"\nPassed: {passed}/{len(results)}")
    print_info(f"Failed: {failed}/{len(results)}")
    
    if failed == 0:
        print_success("\n🎉 All tests passed! Ready for Render deployment.")
        return 0
    else:
        print_error(f"\n❌ {failed} test(s) failed. Check logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
