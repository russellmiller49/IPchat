"""
Unified hybrid search implementation supporting all editions.
Migrated from indexing/hybrid_search.py
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from ipchat.core.config import IPChatConfig, Edition

logger = logging.getLogger(__name__)


class HybridSearcher:
    """
    Unified hybrid search supporting vector (FAISS), keyword (BM25), and optional SQL.
    Works across all editions with appropriate fallbacks.
    """
    
    def __init__(self, config: IPChatConfig):
        self.config = config
        self.faiss_searcher = None
        self.bm25_searcher = None
        self.sql_searcher = None
        
        self._initialize_searchers()
    
    def _initialize_searchers(self):
        """Initialize search components based on edition and config."""
        
        # Initialize FAISS vector search
        if self.config.retrieval.vector_store == "faiss":
            try:
                from .faiss_search import FaissSearcher
                self.faiss_searcher = FaissSearcher(
                    index_path=str(self.config.data.faiss_index),
                    chunks_path=str(self.config.data.chunks_meta)
                )
                logger.info(f"Initialized FAISS searcher with {len(self.faiss_searcher.chunks)} chunks")
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
        
        # Initialize BM25 keyword search
        if self.config.retrieval.keyword_index == "bm25":
            try:
                from .bm25_search import BM25Searcher
                self.bm25_searcher = BM25Searcher(
                    index_path=str(self.config.data.bm25_index)
                )
                logger.info(f"Initialized BM25 searcher")
            except Exception as e:
                logger.warning(f"Failed to initialize BM25: {e}")
        
        # Initialize SQL search (only for full edition)
        if self.config.edition == Edition.FULL and self.config.infra.sql_backend:
            try:
                from .sql_search import SQLSearcher
                self.sql_searcher = SQLSearcher(
                    connection_string=self.config.infra.sql_connection_string
                )
                logger.info("Initialized SQL searcher")
            except Exception as e:
                logger.warning(f"Failed to initialize SQL: {e}")
    
    def search(
        self, 
        query: str, 
        num_results: int = None,
        use_reranker: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across available backends.
        
        Args:
            query: Search query
            num_results: Number of results to return
            use_reranker: Whether to apply reranking
            
        Returns:
            List of search results with scores
        """
        if num_results is None:
            num_results = self.config.retrieval.num_results
        
        results = []
        weights = []
        
        # Collect results from each searcher
        if self.faiss_searcher:
            vector_results = self.faiss_searcher.search(query, num_results * 2)
            results.append(vector_results)
            weights.append(self.config.retrieval.vector_weight)
            logger.debug(f"Vector search returned {len(vector_results)} results")
        
        if self.bm25_searcher:
            keyword_results = self.bm25_searcher.search(query, num_results * 2)
            results.append(keyword_results)
            weights.append(self.config.retrieval.bm25_weight)
            logger.debug(f"BM25 search returned {len(keyword_results)} results")
        
        if self.sql_searcher:
            sql_results = self.sql_searcher.search(query, num_results)
            results.append(sql_results)
            weights.append(self.config.retrieval.sql_weight)
            logger.debug(f"SQL search returned {len(sql_results)} results")
        
        # Normalize weights
        if weights:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        # Combine and rank results
        combined = self._combine_results(results, weights, num_results)
        
        # Apply reranking if configured
        if use_reranker and self.config.retrieval.reranker:
            combined = self._rerank_results(combined, query)
        
        return combined[:num_results]
    
    def _combine_results(
        self, 
        result_sets: List[List[Dict]], 
        weights: List[float],
        num_results: int
    ) -> List[Dict[str, Any]]:
        """
        Combine results from multiple searchers using weighted scoring.
        """
        # Aggregate scores by document ID
        doc_scores = {}
        doc_data = {}
        
        for results, weight in zip(result_sets, weights):
            for result in results:
                doc_id = result.get("chunk_id", result.get("id"))
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_data[doc_id] = result
                
                # Normalize and weight the score
                score = result.get("score", 0.5)
                doc_scores[doc_id] += score * weight
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Build final results
        combined = []
        for doc_id, score in sorted_docs[:num_results]:
            result = doc_data[doc_id].copy()
            result["combined_score"] = score
            result["sources"] = self._identify_sources(doc_id, result_sets)
            combined.append(result)
        
        return combined
    
    def _identify_sources(self, doc_id: str, result_sets: List[List[Dict]]) -> List[str]:
        """Identify which searchers returned this document."""
        sources = []
        source_names = ["vector", "bm25", "sql"]
        
        for results, name in zip(result_sets, source_names[:len(result_sets)]):
            for result in results:
                if result.get("chunk_id", result.get("id")) == doc_id:
                    sources.append(name)
                    break
        
        return sources
    
    def _rerank_results(
        self, 
        results: List[Dict], 
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Apply reranking to improve result relevance.
        Placeholder for future reranker integration.
        """
        # TODO: Integrate with Cohere, CrossEncoder, or custom reranker
        logger.debug("Reranking not yet implemented, returning original order")
        return results


# Convenience function for backwards compatibility
def create_hybrid_searcher(config: Optional[IPChatConfig] = None) -> HybridSearcher:
    """Create a hybrid searcher with default or provided config."""
    if config is None:
        from ipchat.core.config import load_config
        config = load_config()
    return HybridSearcher(config)