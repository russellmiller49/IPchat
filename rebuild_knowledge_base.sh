#!/bin/bash
# Rebuild the knowledge base from extracted documents
set -e

echo "=========================================="
echo "KNOWLEDGE BASE REBUILD"
echo "=========================================="

# Check if we have extractions
if [ ! -d "data/oe_final_outputs" ]; then
    echo "Error: No extractions found in data/oe_final_outputs/"
    echo "Run extraction first: python tools/medical_extractor.py --batch"
    exit 1
fi

NUM_DOCS=$(ls data/oe_final_outputs/*.oe_final.json 2>/dev/null | wc -l)
echo "Found $NUM_DOCS extracted documents"

# Step 1: Chunking
echo ""
echo "Step 1: Chunking documents..."
echo "----------------------------------------"
python3 chunking/chunker.py \
  --trials-dir data/oe_final_outputs \
  --out data/chunks/chunks.jsonl

if [ -f "data/chunks/chunks.jsonl" ]; then
    NUM_CHUNKS=$(wc -l < data/chunks/chunks.jsonl)
    echo "✓ Created $NUM_CHUNKS chunks"
else
    echo "✗ Failed to create chunks"
    exit 1
fi

# Step 2: FAISS Vector Index
echo ""
echo "Step 2: Building FAISS vector index..."
echo "----------------------------------------"
if python3 -c "import sentence_transformers, faiss" 2>/dev/null; then
    python3 indexing/build_faiss.py \
      --chunks data/chunks/chunks.jsonl \
      --out-dir data/index \
      --model intfloat/e5-large-v2
    
    if [ -f "data/index/faiss.index" ]; then
        INDEX_SIZE=$(ls -lh data/index/faiss.index | awk '{print $5}')
        echo "✓ FAISS index created: $INDEX_SIZE"
    else
        echo "✗ Failed to create FAISS index"
    fi
else
    echo "⚠ Skipping FAISS (missing dependencies)"
    echo "  Install with: pip install sentence-transformers faiss-cpu"
fi

# Step 3: BM25 Keyword Index
echo ""
echo "Step 3: Building BM25 keyword index..."
echo "----------------------------------------"
if python3 -c "import rank_bm25" 2>/dev/null; then
    python3 indexing/build_bm25.py \
      --chunks data/chunks/chunks.jsonl \
      --out-dir data/index
    
    if [ -f "data/index/bm25.pkl" ]; then
        BM25_SIZE=$(ls -lh data/index/bm25.pkl | awk '{print $5}')
        echo "✓ BM25 index created: $BM25_SIZE"
    else
        echo "✗ Failed to create BM25 index"
    fi
else
    echo "⚠ Skipping BM25 (missing dependencies)"
    echo "  Install with: pip install rank-bm25"
fi

# Step 4: PostgreSQL (if available)
echo ""
echo "Step 4: Loading to PostgreSQL..."
echo "----------------------------------------"
if [ ! -z "$DATABASE_URL" ]; then
    echo "Database URL found, loading structured data..."
    
    # Load structured data
    python3 ingestion/load_json_to_pg.py \
      --trials-dir data/oe_final_outputs \
      --chapters-dir "Textbooks/Chapter json" 2>/dev/null || true
    
    # Build PostgreSQL full-text search
    python3 indexing/build_bm25_pg.py 2>/dev/null || true
    
    echo "✓ PostgreSQL data loaded"
else
    echo "⚠ Skipping PostgreSQL (DATABASE_URL not set)"
    echo "  To enable: export DATABASE_URL=postgresql://user:pass@localhost:5432/medical_rag"
fi

# Summary
echo ""
echo "=========================================="
echo "KNOWLEDGE BASE REBUILD COMPLETE"
echo "=========================================="
echo "Summary:"
echo "  Documents:  $NUM_DOCS"
echo "  Chunks:     ${NUM_CHUNKS:-0}"
echo "  Indices:    FAISS + BM25"

# Check if search dependencies are ready
echo ""
echo "Next steps:"
if [ -f "data/index/faiss.index" ] && [ -f "data/index/bm25.pkl" ]; then
    echo "  ✓ Knowledge base ready for search!"
    echo ""
    echo "  Start the API:"
    echo "    uvicorn backend.api.main:app --reload"
    echo ""
    echo "  Start the UI:"
    echo "    streamlit run chatbot_app.py"
else
    echo "  ⚠ Install missing dependencies:"
    echo "    pip install -r requirements.txt"
    echo ""
    echo "  Then run this script again."
fi