#!/usr/bin/env python3
"""
FastAPI service for hybrid retrieval and answer composition.

Endpoints:
  POST /search {query, k} → ranked chunks (FAISS-backed for now)
  POST /answer {query} → answer with citations (stub composition)
  GET /facets → basic facets placeholder

Run:
  uvicorn backend.api.main:app --reload --port 8000
"""
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import os

from indexing.search import FaissSearcher
from indexing.hybrid_search import HybridSearcher

INDEX_DIR = Path("data/index")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/e5-small-v2")

app = FastAPI(title="Evidence Chatbot API", version="0.3.0-lite")
searcher: Optional[FaissSearcher] = None
hybrid_searcher: Optional[HybridSearcher] = None


class SearchRequest(BaseModel):
    query: str
    k: int = 10
    use_hybrid: bool = True  # Enable hybrid search by default


class SearchHit(BaseModel):
    score: float
    chunk_id: str
    document_id: str
    source: Optional[str] = None  # vector, bm25, or sql
    text_preview: Optional[str] = None


class SearchResponse(BaseModel):
    hits: List[SearchHit]


class AnswerRequest(BaseModel):
    query: str
    k: int = 10


class AnswerResponse(BaseModel):
    answer: str
    citations: List[str]


def ensure_searcher() -> FaissSearcher:
    global searcher
    if searcher is None:
        searcher = FaissSearcher(INDEX_DIR, model_name=EMBED_MODEL)
    return searcher


def ensure_hybrid_searcher() -> HybridSearcher:
    global hybrid_searcher
    if hybrid_searcher is None:
        hybrid_searcher = HybridSearcher(
            INDEX_DIR, model_name=EMBED_MODEL, database_url=os.getenv("DATABASE_URL")
        )
    return hybrid_searcher


@app.post("/search", response_model=SearchResponse)
def post_search(req: SearchRequest):
    if req.use_hybrid:
        # Use hybrid search
        hs = ensure_hybrid_searcher()
        results = hs.search(req.query, k=req.k)
        hits = [
            SearchHit(
                score=r.score,
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                source=r.source,
                text_preview=r.text[:200] if r.text else None,
            )
            for r in results
        ]
    else:
        # Use vector-only search
        s = ensure_searcher()
        raw = s.search(req.query, k=req.k)
        hits = [
            SearchHit(
                score=score,
                chunk_id=meta["chunk_id"],
                document_id=meta["document_id"],
                source="vector",
            )
            for score, meta in raw
        ]
    return SearchResponse(hits=hits)


@app.post("/answer", response_model=AnswerResponse)
def post_answer(req: AnswerRequest):
    # Use hybrid search for better retrieval
    hs = ensure_hybrid_searcher()
    results = hs.search(req.query, k=req.k)

    # Get unique citations from top results
    cits = []
    seen = set()
    for r in results[:5]:
        if r.document_id not in seen:
            cits.append(r.document_id)
            seen.add(r.document_id)

    # In production: fetch full chunks text to compose grounded answer with strict citations.
    # For now, provide a more informative placeholder
    ans = (
        f"Found {len(results)} relevant results using hybrid search (vector + BM25 + SQL). "
        f"Top result from {results[0].source if results else 'none'} with score {results[0].score:.3f if results else 0}.\n"
        "Wire an LLM to compose an evidence-backed response using the retrieved chunks."
    )
    return AnswerResponse(answer=ans, citations=cits)


@app.get("/facets")
def get_facets():
    # Stub for UI filters; when SQL is wired, pull real facets (interventions, outcomes, timepoints, years).
    return {
        "interventions": [],
        "outcomes": [],
        "timepoints": ["P3M", "P6M", "P12M"],
        "years": [],
    }
