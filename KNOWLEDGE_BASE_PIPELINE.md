# Knowledge Base Pipeline: From Extraction to Search

## Overview
This document describes the complete pipeline for processing extracted medical articles from `data/oe_final_outputs/` into a searchable knowledge base.

## Pipeline Steps

### Step 1: Extraction → oe_final_outputs ✅ COMPLETE
- 292 articles already extracted in `data/oe_final_outputs/`
- Format: `*.oe_final.json` files with structured medical evidence

### Step 2: Chunking → chunks.jsonl
**Purpose**: Break documents into searchable segments (~450 tokens each)

**Command**:
```bash
python chunking/chunker.py \
  --trials-dir data/oe_final_outputs \
  --chapters-dir "Textbooks/Chapter json" \
  --out data/chunks/chunks.jsonl
```

**What it does**:
- Reads all `*.oe_final.json` files from `data/oe_final_outputs/`
- Extracts text from sections (abstract, methods, results, etc.)
- Creates overlapping chunks (450 tokens with 80 token overlap)
- Outputs to `data/chunks/chunks.jsonl`

**Output format** (each line):
```json
{
  "chunk_id": "document_name#0",
  "document_id": "document_name",
  "text": "chunk text content...",
  "source": "trial",
  "pages": [1, 2],
  "section_path": ["results"],
  "trial_signals": {}
}
```

### Step 3: Vector Index Building → FAISS
**Purpose**: Enable semantic search using embeddings

**Command**:
```bash
python indexing/build_faiss.py \
  --chunks data/chunks/chunks.jsonl \
  --out-dir data/index \
  --model intfloat/e5-large-v2
```

**What it does**:
- Loads all chunks from `chunks.jsonl`
- Generates embeddings using sentence-transformers
- Creates FAISS index for fast similarity search
- Saves to `data/index/faiss.index` and `data/index/meta.jsonl`

### Step 4: BM25/Keyword Index → PostgreSQL
**Purpose**: Enable keyword-based search

**Option A - Using PostgreSQL**:
```bash
# First, ensure PostgreSQL is running and database exists
export DATABASE_URL=postgresql://user:pass@localhost:5432/medical_rag

# Create schema
psql $DATABASE_URL < sql/schema.sql

# Load chunks for full-text search
python indexing/build_bm25_pg.py
```

**Option B - Using standalone BM25**:
```bash
python indexing/build_bm25.py \
  --chunks data/chunks/chunks.jsonl \
  --out-dir data/index
```

**Output**:
- `data/index/bm25.pkl` - BM25 index
- `data/index/bm25_meta.jsonl` - Metadata
- `data/index/vocab.txt` - Vocabulary

### Step 5: Structured Data → PostgreSQL
**Purpose**: Enable SQL queries on numerical outcomes

**Command**:
```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/medical_rag

python ingestion/load_json_to_pg.py \
  --trials-dir data/oe_final_outputs \
  --chapters-dir "Textbooks/Chapter json"
```

**What it does**:
- Loads structured data into PostgreSQL tables:
  - `studies` - Study metadata (title, year, journal)
  - `outcomes` - Primary/secondary outcomes with statistics
  - `populations` - Patient demographics
  - `adverse_events` - Safety data
  - `chunks` - Text chunks for full-text search

### Step 6: Start API & Search
**Purpose**: Serve the knowledge base

**Command**:
```bash
# Start FastAPI backend
uvicorn backend.api.main:app --reload --port 8000

# In another terminal, start Streamlit UI
streamlit run chatbot_app.py
```

## Complete Pipeline Script

Create `rebuild_knowledge_base.sh`:
```bash
#!/bin/bash
set -e

echo "Step 1: Chunking documents..."
python chunking/chunker.py \
  --trials-dir data/oe_final_outputs \
  --out data/chunks/chunks.jsonl

echo "Step 2: Building FAISS vector index..."
python indexing/build_faiss.py \
  --chunks data/chunks/chunks.jsonl \
  --out-dir data/index

echo "Step 3: Building BM25 keyword index..."
python indexing/build_bm25.py \
  --chunks data/chunks/chunks.jsonl \
  --out-dir data/index

echo "Step 4: Loading to PostgreSQL (if available)..."
if [ ! -z "$DATABASE_URL" ]; then
  python ingestion/load_json_to_pg.py \
    --trials-dir data/oe_final_outputs
  python indexing/build_bm25_pg.py
else
  echo "Skipping PostgreSQL (DATABASE_URL not set)"
fi

echo "Knowledge base rebuild complete!"
echo "Stats:"
echo "  Chunks: $(wc -l < data/chunks/chunks.jsonl)"
echo "  FAISS index: $(ls -lh data/index/faiss.index | awk '{print $5}')"
echo "  BM25 index: $(ls -lh data/index/bm25.pkl | awk '{print $5}')"
```

## Current Status Check

Run this to see current knowledge base status:
```bash
python tools/check_extraction_status.py
```

Shows:
- 292 extracted documents in `oe_final_outputs/`
- 874 chunks already indexed
- FAISS index built
- BM25 index built

## Adding New Documents

When new documents are extracted:

1. **Extract new document**:
```bash
python tools/medical_extractor.py --single "new_paper.json"
```

2. **Rebuild chunks** (incremental):
```bash
python chunking/chunker.py \
  --trials-dir data/oe_final_outputs \
  --out data/chunks/chunks_new.jsonl

# Append to existing chunks
cat data/chunks/chunks_new.jsonl >> data/chunks/chunks.jsonl
```

3. **Rebuild indices**:
```bash
python indexing/build_faiss.py
python indexing/build_bm25.py
```

## Search Architecture

The system uses **hybrid search** combining three methods:

1. **Vector Search (FAISS)**
   - Semantic similarity using embeddings
   - Good for: conceptual queries, synonyms

2. **Keyword Search (BM25)**
   - Traditional term frequency matching
   - Good for: specific terms, acronyms

3. **SQL Search (PostgreSQL)**
   - Structured queries on outcomes
   - Good for: "FEV1 > 15%", "p-value < 0.05"

**Score fusion**:
```python
final_score = 0.5 * vector_score + 0.3 * bm25_score + 0.2 * sql_score
```

## Troubleshooting

### Issue: "No module named sentence_transformers"
```bash
pip install sentence-transformers faiss-cpu
```

### Issue: "PostgreSQL connection failed"
```bash
# Option 1: Use Docker
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:14

# Option 2: Skip PostgreSQL
# The system works without it, just no SQL queries
```

### Issue: "Chunks already exist"
```bash
# To rebuild from scratch
rm data/chunks/chunks.jsonl
rm data/index/*
# Then run pipeline again
```

## Performance Metrics

With current setup (292 documents):
- Chunking: ~2 minutes
- FAISS indexing: ~1 minute
- BM25 indexing: ~30 seconds
- Search latency: <100ms
- Memory usage: ~500MB

## Next Steps

1. **Monitor extraction completions**:
   - 20 documents still need extraction
   - Run: `python tools/medical_extractor.py --batch`

2. **Optimize chunk size**:
   - Current: 450 tokens
   - Consider: 250-350 for more precise retrieval

3. **Add reranking**:
   - Use cross-encoder for better relevance
   - Improves top-k results quality