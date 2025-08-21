"""
RAG Pipeline orchestration for IPChat.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path

from ipchat.core.config import IPChatConfig, Edition
from ipchat.core.retrieval import create_hybrid_searcher
from ipchat.adapters.llm.openai import create_llm_adapter
from ipchat.core.citations.formatter import format_citations_mla


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates search, retrieval, and generation.
    """
    
    def __init__(self, config: IPChatConfig):
        self.config = config
        self.searcher = create_hybrid_searcher(config)
        self.llm = create_llm_adapter(
            provider=config.llm.provider,
            config={
                "model": config.llm.model,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
            }
        )
        
    def query(
        self,
        question: str,
        num_results: Optional[int] = None,
        use_depth: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the full RAG pipeline.
        
        Args:
            question: User's question
            num_results: Number of search results to retrieve
            use_depth: Whether to use depth mode
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        if num_results is None:
            num_results = self.config.retrieval.num_results
            
        if use_depth is None:
            use_depth = self.config.depth_features
        
        # Search for relevant documents
        search_results = self.searcher.search(question, num_results)
        
        # Generate response with depth mode if enabled
        if use_depth and hasattr(self, '_query_with_depth'):
            response = self._query_with_depth(question, search_results)
        else:
            response = self.llm.generate_with_search(
                question,
                search_results,
                system_prompt=self._get_system_prompt()
            )
        
        # Format citations
        citations = self._extract_citations(search_results)
        bibliography = format_citations_mla(citations) if citations else ""
        
        return {
            "answer": response,
            "citations": citations,
            "bibliography": bibliography,
            "search_results": search_results,
            "metadata": {
                "edition": self.config.edition.value,
                "model": self.config.llm.model,
                "num_results": len(search_results),
                "depth_mode": use_depth
            }
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt based on edition."""
        base_prompt = """You are Bronchmonkey, an expert medical research assistant specializing in interventional pulmonology. 
        Provide evidence-based answers using the search results provided. 
        Always cite sources using (Author Year) format.
        Focus on clinical relevance and practical applications."""
        
        if self.config.edition == Edition.FULL:
            base_prompt += "\nYou have access to comprehensive medical databases and full-text articles."
        elif self.config.edition == Edition.SPACE:
            base_prompt += "\nProvide concise, focused answers suitable for quick reference."
            
        return base_prompt
    
    def _extract_citations(self, search_results: List[Dict]) -> List[Dict[str, str]]:
        """Extract citation information from search results."""
        citations = []
        seen = set()
        
        for result in search_results:
            # Try to extract citation info from the result
            source = result.get("source", result.get("chunk_id", ""))
            
            # Skip if we've seen this source
            if source in seen:
                continue
            seen.add(source)
            
            # Parse author and year from source if possible
            # This is a simplified version - you may want to enhance this
            citation = {
                "source": source,
                "text": result.get("text_preview", "")[:200]
            }
            
            # Try to extract author and year
            import re
            match = re.search(r"([A-Za-z]+)[^0-9]*(\d{4})", source)
            if match:
                citation["author"] = match.group(1)
                citation["year"] = match.group(2)
            
            citations.append(citation)
        
        return citations
    
    def _query_with_depth(self, question: str, initial_results: List[Dict]) -> str:
        """
        Enhanced query processing with depth mode.
        This is a placeholder for the full depth mode implementation.
        """
        # TODO: Integrate the full depth mode from utils/depth_mode.py
        # For now, just use enhanced prompting
        
        depth_prompt = """Provide a comprehensive, nuanced analysis that:
        1. Synthesizes evidence from multiple sources
        2. Highlights areas of agreement and disagreement in the literature
        3. Discusses clinical implications and practical applications
        4. Identifies gaps in current knowledge
        5. Suggests areas for future research"""
        
        return self.llm.generate_with_search(
            question,
            initial_results,
            system_prompt=self._get_system_prompt() + "\n\n" + depth_prompt
        )


def create_rag_pipeline(config: Optional[IPChatConfig] = None) -> RAGPipeline:
    """
    Factory function to create a RAG pipeline.
    
    Args:
        config: Optional configuration object
        
    Returns:
        Configured RAG pipeline
    """
    if config is None:
        from ipchat.core.config import load_config
        config = load_config()
    
    return RAGPipeline(config)


__all__ = ["RAGPipeline", "create_rag_pipeline"]