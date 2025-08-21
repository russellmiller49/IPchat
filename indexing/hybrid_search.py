#!/usr/bin/env python3
"""
Hybrid search implementation combining:
1. Vector search (FAISS)
2. Keyword search (BM25)
3. Structured SQL queries
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import re

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    score: float
    source: str  # 'vector', 'bm25', or 'sql'
    text: Optional[str] = None
    metadata: Optional[Dict] = None


class HybridSearcher:
    def __init__(
        self, 
        index_dir: Path | str, 
        model_name: str = "intfloat/e5-large-v2",
        database_url: Optional[str] = None
    ):
        from sentence_transformers import SentenceTransformer
        import faiss
        
        self.index_dir = Path(index_dir)
        
        # Load vector search components
        self.model = SentenceTransformer(model_name)
        self.faiss = faiss.read_index(str(self.index_dir / "faiss.index"))
        self.faiss_meta: List[Dict] = []
        with (self.index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.faiss_meta.append(json.loads(line))
        
        # Load BM25 components
        with open(self.index_dir / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        
        self.bm25_meta: List[Dict] = []
        with (self.index_dir / "bm25_meta.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.bm25_meta.append(json.loads(line))
        
        # Load chunks for text retrieval
        self.chunks = {}
        chunks_file = Path("data/chunks/chunks.jsonl")
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    self.chunks[chunk['chunk_id']] = chunk
        
        # Database connection
        self.database_url = database_url or os.getenv("DATABASE_URL")
    
    def preprocess_query(self, query: str) -> List[str]:
        """Preprocess query for BM25 (same as indexing)."""
        # Inline preprocessing to avoid import issues
        text = query.lower()
        text = re.sub(r'[^\w\s\-\%\.]', ' ', text)
        tokens = text.split()
        
        # Stopwords
        stopwords = set([
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
            'was', 'will', 'would', 'could', 'should', 'may', 'might', 'must'
        ])
        
        medical_keep = {
            'no', 'not', 'with', 'without', 'after', 'before', 'during', 'between',
            'versus', 'vs', 'compared', 'than', 'more', 'less', 'greater', 'fewer'
        }
        
        filtered_tokens = []
        for token in tokens:
            if (token in medical_keep or 
                any(c.isdigit() for c in token) or 
                token not in stopwords):
                if len(token) > 1:
                    filtered_tokens.append(token)
        
        return filtered_tokens
    
    def vector_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform vector similarity search using FAISS."""
        # Encode query
        qv = self.model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv, dtype="float32")
        
        # Search
        scores, idxs = self.faiss.search(qv, k)
        
        results = []
        for rank in range(min(k, len(idxs[0]))):
            idx = int(idxs[0][rank])
            meta = self.faiss_meta[idx]
            results.append(SearchResult(
                chunk_id=meta['chunk_id'],
                document_id=meta['document_id'],
                score=float(scores[0][rank]),
                source='vector'
            ))
        
        return results
    
    def bm25_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Perform keyword search using BM25."""
        # Preprocess query
        query_tokens = self.preprocess_query(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                meta = self.bm25_meta[idx]
                results.append(SearchResult(
                    chunk_id=meta['chunk_id'],
                    document_id=meta['document_id'],
                    score=float(scores[idx]),
                    source='bm25'
                ))
        
        return results
    
    def sql_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Perform structured SQL search for specific patterns.
        Detects query intent and searches appropriate tables.
        """
        if not self.database_url:
            return []
        
        results = []
        query_lower = query.lower()
        
        try:
            # Parse connection string
            db_parts = self.database_url.replace("postgresql+psycopg2://", "postgresql://")
            conn = psycopg2.connect(db_parts, cursor_factory=RealDictCursor)
            
            with conn.cursor() as cur:
                # Detect query patterns and search accordingly
                
                # Pattern 1: Pneumothorax or adverse events
                if any(term in query_lower for term in ['pneumothorax', 'adverse', 'safety', 'complication']):
                    cur.execute("""
                        SELECT DISTINCT s.study_id, s.title, sa.pt, sa.percentage
                        FROM studies s
                        LEFT JOIN safety sa ON s.study_id = sa.study_id
                        WHERE s.title ILIKE %s 
                        OR sa.pt ILIKE %s
                        LIMIT %s
                    """, (f'%{query}%', f'%{query}%', k))
                    
                    for row in cur.fetchall():
                        if row['study_id']:
                            results.append(SearchResult(
                                chunk_id=f"{row['study_id']}#sql",
                                document_id=row['study_id'],
                                score=1.0,  # SQL results don't have natural scores
                                source='sql',
                                metadata={'title': row['title'], 'event': row.get('pt'), 'percentage': row.get('percentage')}
                            ))
                
                # Pattern 2: FEV1 or outcomes
                elif any(term in query_lower for term in ['fev1', 'outcome', 'improvement', 'lung function']):
                    cur.execute("""
                        SELECT DISTINCT s.study_id, s.title, o.name, o.est, o.unit, o.timepoint_iso8601
                        FROM studies s
                        LEFT JOIN outcomes o ON s.study_id = o.study_id
                        WHERE s.title ILIKE %s 
                        OR o.name ILIKE %s
                        LIMIT %s
                    """, (f'%{query}%', f'%{query}%', k))
                    
                    for row in cur.fetchall():
                        if row['study_id']:
                            results.append(SearchResult(
                                chunk_id=f"{row['study_id']}#sql",
                                document_id=row['study_id'],
                                score=1.0,
                                source='sql',
                                metadata={
                                    'title': row['title'], 
                                    'outcome': row.get('name'),
                                    'value': row.get('est'),
                                    'unit': row.get('unit'),
                                    'timepoint': row.get('timepoint_iso8601')
                                }
                            ))
                
                # Pattern 3: Specific interventions (BLVR, valve, etc.)
                elif any(term in query_lower for term in ['blvr', 'valve', 'endobronchial', 'zephyr', 'spiration']):
                    cur.execute("""
                        SELECT DISTINCT study_id, title
                        FROM studies
                        WHERE title ILIKE %s
                        LIMIT %s
                    """, (f'%{query}%', k))
                    
                    for row in cur.fetchall():
                        results.append(SearchResult(
                            chunk_id=f"{row['study_id']}#sql",
                            document_id=row['study_id'],
                            score=1.0,
                            source='sql',
                            metadata={'title': row['title']}
                        ))
            
            conn.close()
            
        except Exception as e:
            print(f"SQL search error: {e}")
        
        return results
    
    def normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores across different search methods."""
        if not results:
            return results
        
        # Group by source
        by_source = {}
        for r in results:
            if r.source not in by_source:
                by_source[r.source] = []
            by_source[r.source].append(r)
        
        # Normalize each source's scores to 0-1 range
        normalized = []
        for source, items in by_source.items():
            if not items:
                continue
            
            scores = [r.score for r in items]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            range_score = max_score - min_score if max_score != min_score else 1.0
            
            for r in items:
                # Normalize to 0-1
                if source == 'sql':
                    # SQL results are binary, give them moderate weight
                    r.score = 0.7
                else:
                    r.score = (r.score - min_score) / range_score if range_score > 0 else 0.5
                normalized.append(r)
        
        return normalized
    
    def fuse_results(
        self, 
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
        sql_results: List[SearchResult],
        weights: Dict[str, float] = None
    ) -> List[SearchResult]:
        """
        Fuse results from different search methods using weighted reciprocal rank fusion.
        """
        if weights is None:
            weights = {
                'vector': 0.5,   # Semantic similarity
                'bm25': 0.3,     # Keyword matching  
                'sql': 0.2       # Structured data
            }
        
        # Normalize scores
        all_results = (
            self.normalize_scores(vector_results) +
            self.normalize_scores(bm25_results) +
            self.normalize_scores(sql_results)
        )
        
        # Aggregate scores by chunk_id
        chunk_scores = {}
        chunk_data = {}
        
        for r in all_results:
            if r.chunk_id not in chunk_scores:
                chunk_scores[r.chunk_id] = 0
                chunk_data[r.chunk_id] = r
            
            # Apply weight based on source
            weighted_score = r.score * weights.get(r.source, 0.5)
            chunk_scores[r.chunk_id] += weighted_score
        
        # Sort by combined score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final results
        final_results = []
        for chunk_id, score in sorted_chunks:
            result = chunk_data[chunk_id]
            result.score = score
            
            # Add text if available
            if chunk_id in self.chunks:
                result.text = self.chunks[chunk_id].get('text', '')
            
            final_results.append(result)
        
        return final_results
    
    def search(
        self, 
        query: str, 
        k: int = 10,
        use_vector: bool = True,
        use_bm25: bool = True,
        use_sql: bool = True,
        weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining multiple methods.
        
        Args:
            query: Search query
            k: Number of results per method
            use_vector: Enable vector search
            use_bm25: Enable BM25 keyword search
            use_sql: Enable SQL structured search
            weights: Custom weights for score fusion
        
        Returns:
            List of SearchResult objects sorted by combined score
        """
        # Perform searches
        vector_results = self.vector_search(query, k) if use_vector else []
        bm25_results = self.bm25_search(query, k) if use_bm25 else []
        sql_results = self.sql_search(query, k) if use_sql else []
        
        # Fuse results
        fused = self.fuse_results(vector_results, bm25_results, sql_results, weights)
        
        # Return top k
        return fused[:k]


def test_hybrid_search():
    """Test the hybrid search functionality."""
    searcher = HybridSearcher("data/index")
    
    queries = [
        "what percent of patients with BLVR had pneumothorax",
        "FEV1 improvement at 12 months",
        "endobronchial valve complications",
        "Zephyr valve outcomes"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        results = searcher.search(query, k=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result.score:.3f} | Source: {result.source}")
            print(f"   Document: {result.document_id}")
            print(f"   Chunk: {result.chunk_id}")
            if result.metadata:
                print(f"   Metadata: {result.metadata}")
            if result.text:
                print(f"   Preview: {result.text[:150]}...")


if __name__ == "__main__":
    test_hybrid_search()