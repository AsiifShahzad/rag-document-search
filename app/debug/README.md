# RAG Pipeline Debug Tools

Complete debugging and diagnostics tools for your DocumentChat RAG backend.

## Overview

This package provides three comprehensive debugging solutions:

1. **Verbose Console Logging** - Detailed logs printed during pipeline execution
2. **Standalone Debug Script** - Run offline verification tests
3. **Debug API Endpoints** - Query debugging information via REST API

---

## 1. VERBOSE CONSOLE LOGGING

All RAG pipeline stages now log detailed information to the console.

### What Gets Logged

**Document Upload** (`/upload` endpoint):
```
[INGESTION] Starting document ingestion
[INGESTION] Loaded N pages from PDF
[INGESTION] Split into N chunks
  Chunk X: Y characters | Page: Z
[INGESTION] Generated N embeddings
  Embedding dimensions: X
  Sample embedding: [...]
  Embedding norm: X.XXXX
[INGESTION] Vectors stored successfully
```

**Query Processing** (`/ask` endpoint):
```
[API] Received /ask request
[API] Question: "..."
[RETRIEVER] Generating embedding for query: "..."
[RETRIEVER] Embedding generated - dimensions: X
[RETRIEVER] Searching Pinecone for top 20 similar chunks...
[RETRIEVER] Retrieved X results from Pinecone
  [1] Score: X.XXXX | Source: ... | Page: X
  [2] Score: X.XXXX | ...
[PIPELINE] Step 3: Reranking top candidates...
[PROMPT_BUILDER] Building prompt with N chunks
[GENERATOR] Attempting model: llama-3.3-70b-versatile
[GENERATOR] ✓ Model succeeded
[API] /ask request complete
```

### How to View Logs

1. **Terminal Output**: Watch your backend terminal while making API requests
2. **Log Redirection**: Redirect stdout to a file:
   ```bash
   python -m uvicorn app.main:app >> logs.txt 2>&1
   ```

---

## 2. STANDALONE DEBUG SCRIPT

Run comprehensive verification tests without making API calls.

### Usage

```bash
python app/debug/pipeline_verification.py
```

### What It Tests

**Phase 1: Embedding Model Verification**
- Loads embedding model
- Generates test embedding
- Checks:
  - Correct dimensions (384)
  - Non-zero vectors
  - No NaN values
  - Proper normalization

**Phase 2: Pinecone Connection & Data**
- Connects to Pinecone
- Retrieves index statistics
- Checks:
  - Connection status
  - Total vector count
  - Sample vectors with metadata
  - Warnings if no vectors found

**Phase 3: Retrieval & Similarity Search**
- Tests multiple queries
- Performs vector similarity search
- Checks:
  - Result counts
  - Similarity score ranges
  - Retrieved content quality
  - Average/min/max scores

**Phase 4: Reranking**
- Loads reranker model
- Tests reranking with sample pairs
- Verifies scores are reasonable

**Phase 5: End-to-End Pipeline**
- Runs complete retrieval pipeline
- Tests actual queries
- Verifies final results

### Expected Output

```
================================================================================
RAG PIPELINE DEBUG REPORT
================================================================================

================================================================================
PHASE: Embedding Model
================================================================================
status: ✓ Working
dimensions: 384
embedding_norm: 25.123456
has_zeros: False
has_nans: False

...

================================================================================
IDENTIFIED ISSUES
================================================================================

If healthy:
✅ NO ISSUES FOUND - Pipeline appears healthy!

If problems found:
🔴 CRITICAL ISSUES:
  [Phase Name] Error description

🟡 WARNINGS:
  [Phase Name] Warning description
```

### Interpreting Results

**✓ All checks passed**: Pipeline is working correctly
**⚠️ Warnings**: Non-fatal issues, may affect quality
**🔴 Critical Issues**: Blocking problems preventing operation

---

## 3. DEBUG API ENDPOINTS

Query detailed diagnostic information via HTTP.

### Endpoints

#### `/debug/health-detailed`
Detailed health check with component diagnostics.

```bash
curl http://localhost:8000/debug/health-detailed
```

Response:
```json
{
  "status": "ok",
  "timestamp": "2026-03-12T...",
  "results": {
    "components": {
      "embedding_model": {
        "status": "ok",
        "dimensions": 384,
        "norm": 25.123
      },
      "pinecone": {
        "status": "ok",
        "total_vectors": 1250,
        "index_name": "bge-small-index"
      },
      "reranker": {
        "status": "ok",
        "model": "BAAI/bge-reranker-base"
      }
    },
    "overall": "ok"
  }
}
```

#### `/debug/pinecone-stats`
Pinecone index statistics and sample vectors.

```bash
curl http://localhost:8000/debug/pinecone-stats
```

#### `/debug/test-embedding?query=<query>`
Test embedding generation for a specific query.

```bash
curl "http://localhost:8000/debug/test-embedding?query=What%20are%20masked%20autoencoders"
```

Response shows:
- Embedding dimensions
- Vector norm, mean, std
- Sample values

#### `/debug/test-retrieval?query=<query>&top_k=10`
Test vector similarity search.

```bash
curl "http://localhost:8000/debug/test-retrieval?query=masked%20autoencoders&top_k=10"
```

Response shows:
- Number of results
- Similarity scores (avg, min, max)
- Retrieved chunks with:
  - Similarity score
  - Source and page
  - Text preview

#### `/debug/test-reranking?query=<query>&top_k=5`
Test full retrieval + reranking pipeline.

```bash
curl "http://localhost:8000/debug/test-reranking?query=autoencoders&top_k=5"
```

Response shows:
- Raw retrieval count
- Reranked results with:
  - Rerank score
  - Vector score
  - Source/page
  - Text preview

#### `/debug/test-pipeline?query=<query>`
Test complete RAG pipeline.

```bash
curl "http://localhost:8000/debug/test-pipeline?query=What%20is%20this%20paper%20about"
```

#### `/debug/verify-all?test_query=<optional-query>`
Run comprehensive verification of all components.

```bash
curl "http://localhost:8000/debug/verify-all?test_query=masked%20autoencoders"
```

Response shows:
- Test results for each component
- Overall pipeline status
- Pass/fail/warning indicators

#### `/debug/logs-sample`
Information about available logging.

```bash
curl http://localhost:8000/debug/logs-sample
```

---

## DEBUGGING WORKFLOW

### Step 1: Run Standalone Script
```bash
python app/debug/pipeline_verification.py
```
Let this complete and note any critical issues.

### Step 2: Check Console Logs
Make a request and watch the console output:
```bash
# Upload a document
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload

# Ask a question
curl -X POST -H "Content-Type: application/json" \
  -d '{"question":"What are masked autoencoders?"}' \
  http://localhost:8000/ask
```

### Step 3: Use Debug API Endpoints
Get structured diagnostic data:
```bash
# Check overall health
curl http://localhost:8000/debug/verify-all

# Test retrieval for a query
curl "http://localhost:8000/debug/test-retrieval?query=your%20query"

# Test reranking
curl "http://localhost:8000/debug/test-reranking?query=your%20query"
```

### Step 4: Interpret Results

**If `/ask` always returns "I couldn't find relevant information":**

1. Run `/debug/verify-all` - Check what fails
2. Check `/debug/pinecone-stats` - Verify vectors exist
3. Run `/debug/test-retrieval?query=<your-query>` - See what's retrieved
4. Watch console logs during `/ask` request - See full pipeline trace

**If retrieval returns low scores:**
- Embedding model may not understand the document domain
- Consider re-embedding with a domain-specific model

**If no vectors found:**
- Check that documents uploaded successfully
- Check `/upload` endpoint logs
- Verify `/debug/pinecone-stats` shows vectors

**If reranking scores don't improve results:**
- May indicate bad initial retrieval
- Check raw vector similarity scores first

---

## COMMON ISSUES & SOLUTIONS

### Issue: "No relevant information found" for ALL queries

**Check:**
1. Is `total_vectors` > 0 in `/debug/pinecone-stats`?
   - If no: Documents not ingested, check `/upload` logs
   - If yes: Continue to step 2

2. Do `/debug/test-retrieval` results exist with good scores?
   - If no: Embedding model issue or poor similarity
   - If yes: Continue to step 3

3. Does `/debug/test-pipeline` return chunks?
   - If no: Reranking eliminating all results
   - If yes: LLM is receiving context but not answering

4. Check LLM logs for errors or timeouts

### Issue: Retrieved chunks are irrelevant

**Causes:**
- Embedding model doesn't understand document domain
- Query is too different from document content
- Document chunking is creating bad chunks

**Fix:**
- Check chunk quality in ingestion logs
- Verify similarity scores in `/debug/test-retrieval`
- Consider different embedding model

### Issue: Reranker not helping

**Check:**
- Are raw vector scores already low?
- Is reranking score different from vector score?
- Try a different `top_k` value

### Issue: LLM timeouts

**Fix:**
- Reduce context size in prompt_builder.py
- Reduce `top_k` in pipeline retrieval
- Check LLM API status

---

## PRODUCTION CONSIDERATIONS

Before deploying to production:

1. **Disable debug endpoints** in `app/debug/debug_routes.py`:
   ```python
   # Remove or conditionally include based on environment
   if os.getenv("ENVIRONMENT") == "development":
       router.include_router(debug_router)
   ```

2. **Reduce verbose logging** by adjusting print statements in production

3. **Secure logs** - Don't expose prompts or context in production

4. **Monitor performance** - Track:
   - Embedding generation time
   - Pinecone query time
   - Reranking time
   - LLM response time
   - Total `/ask` latency

---

## SUPPORT

If you encounter issues:

1. Run the standalone verification script
2. Check all debug endpoints
3. Review the console logs with detailed trace
4. Share the verification report output
