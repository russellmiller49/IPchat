"""
IPChat - Bronchmonkey Interventional Pulmonology Research Assistant

A unified package supporting multiple deployment editions:
- full: Complete system with PostgreSQL, API server, and all features
- lite: Lightweight version with local indexes, no database
- space: Hugging Face Space deployment with pre-built indexes
"""

__version__ = "0.2.0"
__author__ = "Russell Miller"

from ipchat.core.config import load_config, Edition

__all__ = ["load_config", "Edition"]