#!/usr/bin/env bash
set -euo pipefail
: "${EMBED_MODEL:=intfloat/e5-small-v2}"
: "${CHUNK_SIZE:=450}"
: "${CHUNK_OVERLAP:=80}"

echo ">> Chunking with size=$CHUNK_SIZE overlap=$CHUNK_OVERLAP"
python chunking/chunker.py --trials-dir data/complete_extractions --chapters-dir Textbooks --out data/chunks/chunks.jsonl

echo ">> Building FAISS with EMBED_MODEL=$EMBED_MODEL"
python indexing/build_faiss.py --chunks data/chunks/chunks.jsonl --out-dir data/index

echo ">> Done. Index at data/index/ and chunks at data/chunks/chunks.jsonl"