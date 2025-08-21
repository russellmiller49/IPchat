"""
Configuration management for IPChat editions.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


class Edition(str, Enum):
    """Available deployment editions."""
    FULL = "full"
    LITE = "lite"
    SPACE = "space"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.3
    max_tokens: int = 4096
    depth_model: Optional[str] = "gpt-5-2025-08-07"  # For depth mode
    

class RetrievalConfig(BaseModel):
    """Retrieval system configuration."""
    vector_store: str = "faiss"
    keyword_index: str = "bm25"
    reranker: Optional[str] = None
    num_results: int = 10
    vector_weight: float = 0.5
    bm25_weight: float = 0.3
    sql_weight: float = 0.2
    

class DataConfig(BaseModel):
    """Data paths configuration."""
    root: Path = Path("./data")
    faiss_index: Optional[Path] = None
    bm25_index: Optional[Path] = None
    chunks_meta: Optional[Path] = None
    oe_jsons_dir: Optional[Path] = None
    
    @validator("faiss_index", "bm25_index", "chunks_meta", "oe_jsons_dir", pre=True, always=True)
    def set_default_paths(cls, v, values):
        if v is None and "root" in values:
            root = values["root"]
            field_name = cls.__fields__[v].name if hasattr(cls, "__fields__") else None
            if field_name == "faiss_index":
                return root / "index" / "faiss.index"
            elif field_name == "bm25_index":
                return root / "index" / "bm25.pkl"
            elif field_name == "chunks_meta":
                return root / "index" / "meta.jsonl"
            elif field_name == "oe_jsons_dir":
                return root / "oe_final_outputs"
        return v


class InfraConfig(BaseModel):
    """Infrastructure configuration."""
    sql_backend: Optional[str] = None  # "postgres" for full, None for lite/space
    sql_connection_string: Optional[str] = None
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    ui_host: str = "0.0.0.0"
    ui_port: int = 8501  # 7860 for space
    

class AuthConfig(BaseModel):
    """Authentication configuration."""
    enabled: bool = False
    basic_users_env: Optional[str] = "BASIC_AUTH_USERS"
    session_secret_env: Optional[str] = "SESSION_SECRET"


class IPChatConfig(BaseSettings):
    """Main configuration for IPChat."""
    edition: Edition = Edition.LITE
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    infra: InfraConfig = Field(default_factory=InfraConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    
    # Feature flags
    depth_features: bool = True
    debug_mode: bool = False
    
    class Config:
        env_prefix = "IPCHAT_"
        env_nested_delimiter = "__"
        env_file = ".env"
        env_file_encoding = "utf-8"
        
    @validator("edition", pre=True)
    def validate_edition(cls, v):
        if isinstance(v, str):
            return Edition(v.lower())
        return v
        
    def model_post_init(self, __context):
        """Apply edition-specific defaults after initialization."""
        self.apply_edition_defaults()
        
    def apply_edition_defaults(self):
        """Apply defaults based on edition."""
        if self.edition == Edition.FULL:
            self.infra.sql_backend = self.infra.sql_backend or "postgres"
            self.infra.sql_connection_string = self.infra.sql_connection_string or \
                "postgresql://rm:@localhost/medical_rag"
            self.auth.enabled = True
            
        elif self.edition == Edition.LITE:
            self.infra.sql_backend = None
            self.auth.enabled = False
            
        elif self.edition == Edition.SPACE:
            self.infra.sql_backend = None
            self.infra.ui_port = 7860
            self.auth.enabled = True
            self.data.root = Path("/app/data")  # Space container path


def load_config(
    edition: Optional[str] = None,
    config_file: Optional[Path] = None,
    **overrides
) -> IPChatConfig:
    """
    Load configuration for the specified edition.
    
    Args:
        edition: Edition name (full/lite/space) or from env IPCHAT_EDITION
        config_file: Optional config file path
        **overrides: Additional config overrides
    
    Returns:
        IPChatConfig instance
    """
    # Check environment for edition
    if edition is None:
        edition = os.getenv("IPCHAT_EDITION", "lite")
    
    # Create config with edition
    config_dict = {"edition": edition}
    config_dict.update(overrides)
    
    # Load from file if provided
    if config_file and config_file.exists():
        import json
        with open(config_file) as f:
            file_config = json.load(f)
        config_dict.update(file_config)
    
    return IPChatConfig(**config_dict)


# Export main classes
__all__ = ["Edition", "IPChatConfig", "load_config"]