"""
Simplified configuration for the streamlined pipeline.
"""

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

@dataclass
class ExtractionConfig:
    """Configuration for extraction pipeline"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 2000
    batch_size: int = 5

@dataclass
class ChunkingConfig:
    """Configuration for chunking"""
    target_size: int = 400
    overlap: int = 50
    min_size: int = 100
    max_size: int = 600
    chunking_strategy: str = "semantic"  # "semantic" or "fixed"

@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    num_results: int = 10
    vector_weight: float = 0.5
    bm25_weight: float = 0.3
    sql_weight: float = 0.2
    rerank: bool = True

@dataclass
class SimplifiedConfig:
    """Main configuration"""
    extraction: ExtractionConfig = ExtractionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    
    # Paths
    data_dir: Path = Path("data")
    extracted_dir: Path = Path("data/extracted")
    chunks_dir: Path = Path("data/chunks")
    benchmarks_dir: Path = Path("data/benchmarks")
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables"""
        config = cls()
        
        # Override with env vars if present
        if os.getenv("IPCHAT_MODEL"):
            config.extraction.model = os.getenv("IPCHAT_MODEL")
        if os.getenv("IPCHAT_CHUNK_SIZE"):
            config.chunking.target_size = int(os.getenv("IPCHAT_CHUNK_SIZE"))
        if os.getenv("IPCHAT_NUM_RESULTS"):
            config.retrieval.num_results = int(os.getenv("IPCHAT_NUM_RESULTS"))
        
        return config
    
    def validate(self):
        """Validate configuration"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.extracted_dir, self.chunks_dir, self.benchmarks_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return True