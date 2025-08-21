"""
Embedding generation for chunks.
"""

from typing import List, Dict, Any
import openai
import numpy as np
from dataclasses import dataclass

@dataclass
class EmbeddedChunk:
    """Chunk with embedding vector"""
    chunk_id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]

class ChunkEmbedder:
    """Generate embeddings for document chunks"""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = openai.OpenAI()
    
    def embed_chunks(self, chunks: List[Any]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of EmbeddedChunk objects
        """
        embedded_chunks = []
        
        for chunk in chunks:
            # Generate embedding
            response = self.client.embeddings.create(
                model=self.model,
                input=chunk.content
            )
            
            embedding = np.array(response.data[0].embedding)
            
            embedded = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata
            )
            
            embedded_chunks.append(embedded)
        
        return embedded_chunks
    
    def batch_embed(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """
        Batch embed multiple texts efficiently.
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to embed at once
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                model=self.model,
                input=batch
            )
            
            batch_embeddings = [np.array(data.embedding) for data in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings