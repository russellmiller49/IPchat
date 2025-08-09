#!/usr/bin/env python3
"""
Build a FAISS index from chunks.jsonl

Requires:
  pip install sentence-transformers faiss-cpu

Usage:
  python indexing/build_faiss.py \
      --chunks data/chunks/chunks.jsonl \
      --out-dir data/index \
      --model intfloat/e5-large-v2
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np


def load_chunks(path: Path) -> List[Dict]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/chunks.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--model", type=str, default="intfloat/e5-large-v2")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks(args.chunks)
    texts = [c.get("text", "") for c in chunks]

    from sentence_transformers import SentenceTransformer
    import faiss

    model = SentenceTransformer(args.model)
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
    vecs = np.asarray(vecs, dtype="float32")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # Persist index and metadata
    faiss.write_index(index, str(args.out_dir / "faiss.index"))
    with (args.out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"], "document_id": c["document_id"]}) + "\n")

    print(f"Indexed {len(chunks)} chunks → {args.out_dir}")


if __name__ == "__main__":
    main()

