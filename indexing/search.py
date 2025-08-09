"""
Search utilities: vector-only (FAISS) and hybrid placeholder.

Usage:
  from indexing.search import FaissSearcher
  s = FaissSearcher("data/index")
  hits = s.search("Does EBV improve FEV1 at 12 months?", k=10)
"""

from pathlib import Path
from typing import List, Dict, Tuple
import json
import numpy as np


class FaissSearcher:
    def __init__(self, index_dir: Path | str, model_name: str = "intfloat/e5-large-v2"):
        from sentence_transformers import SentenceTransformer
        import faiss

        self.index_dir = Path(index_dir)
        self.model = SentenceTransformer(model_name)
        self.faiss = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.meta: List[Dict] = []
        with (self.index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.meta.append(json.loads(line))

    def search(self, query: str, k: int = 10) -> List[Tuple[float, Dict]]:
        qv = self.model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        scores, idxs = self.faiss.search(qv, k)
        out = []
        for rank in range(min(k, len(idxs[0]))):
            i = int(idxs[0][rank])
            out.append((float(scores[0][rank]), self.meta[i]))
        return out


def hybrid_search(query: str, k: int = 10):
    """
    Placeholder stub for hybrid search. Wire BM25 + SQL as needed.
    """
    raise NotImplementedError("Add BM25/OpenSearch + SQL filters and reranker")

