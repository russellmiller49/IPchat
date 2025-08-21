"""
OpenAI LLM adapter for IPChat.
Wraps the existing openai_client with a unified interface.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import the existing OpenAI client
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ipchat.core.utils.openai_client import get_openai_client


class OpenAIAdapter:
    """
    Adapter for OpenAI models, providing a unified interface.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.client = get_openai_client()
        self.model = self.config.get("model", "gpt-4o-mini")
        self.temperature = self.config.get("temperature", 0.3)
        self.max_tokens = self.config.get("max_tokens", 4096)
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using OpenAI.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text response
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=kwargs.get("model", self.model),
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        return response.choices[0].message.content
    
    def generate_with_search(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response based on search results.
        
        Args:
            query: User query
            search_results: List of search results
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response with citations
        """
        # Format search results as context
        context_parts = []
        for i, result in enumerate(search_results, 1):
            text = result.get("text", result.get("text_preview", ""))
            source = result.get("source", result.get("chunk_id", f"Source {i}"))
            context_parts.append(f"[{i}] {source}:\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Build the prompt
        prompt = f"""Based on the following search results, answer this question: {query}

Search Results:
{context}

Please provide a comprehensive answer based on the evidence above. Include citations in (Author Year) format where appropriate."""
        
        return self.generate(prompt, system_prompt, **kwargs)


def create_llm_adapter(provider: str = "openai", config: Optional[Dict] = None):
    """
    Factory function to create LLM adapters.
    
    Args:
        provider: LLM provider name
        config: Provider-specific configuration
        
    Returns:
        LLM adapter instance
    """
    if provider == "openai":
        return OpenAIAdapter(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")