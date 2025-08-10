# Migration Guide: Unified IPChat Architecture

## Overview

This guide helps you migrate from the old multi-branch structure to the new unified `ipchat` package that supports three editions: **full**, **lite**, and **space**.

## What Changed

### Old Structure (Before)
```
├── main branch/              # Full version with PostgreSQL
│   ├── chatbot_app.py
│   ├── backend/api/main.py
│   └── indexing/*.py
├── lite-perf branch/         # Lightweight version
│   └── (similar structure)
└── bronchmonkey-space/       # Separate HF Space repo
    └── (standalone app)
```

### New Structure (After)
```
├── ipchat/                   # Unified package
│   ├── core/                 # Shared logic (all editions)
│   ├── apps/                 # Edition-specific apps
│   └── cli.py               # Unified CLI
├── pyproject.toml           # Single package definition
└── docker/                  # All Docker configs
```

## Migration Steps

### 1. Install the New Package

```bash
# Clone and checkout the unified branch
git checkout unify-editions

# Install based on your edition needs
pip install -e ".[full]"   # Full edition with all features
pip install -e ".[lite]"   # Lightweight edition
pip install -e ".[space]"  # Hugging Face Space edition
```

### 2. Update Your Environment Variables

The new system uses prefixed environment variables:

```bash
# Old
export OPENAI_API_KEY=sk-...

# New (additional)
export IPCHAT_EDITION=lite  # or full/space
export IPCHAT_DEPTH_FEATURES=true
```

### 3. File Path Mappings

| Old Path | New Path |
|----------|----------|
| `chatbot_app.py` | `ipchat/apps/streamlit_{edition}.py` |
| `backend/api/main.py` | `ipchat/api/server.py` |
| `indexing/hybrid_search.py` | `ipchat/core/retrieval/hybrid_search.py` |
| `indexing/search.py` | `ipchat/core/retrieval/faiss_search.py` |
| `utils/citations.py` | `ipchat/core/citations/formatter.py` |
| `utils/depth_mode.py` | `ipchat/core/pipelines/depth.py` |
| `utils/openai_client.py` | `ipchat/adapters/llm/openai.py` |

### 4. Running the Application

#### Old Way
```bash
# Main branch
streamlit run chatbot_app.py
python -m uvicorn backend.api.main:app

# Lite branch
streamlit run chatbot_app.py  # with different config

# Space
streamlit run chatbot_app.py --server.port 7860
```

#### New Way (Unified CLI)
```bash
# Run any edition with one command
ipchat run --edition full   # Full stack with DB
ipchat run --edition lite   # Lightweight, no DB
ipchat run --edition space  # HF Space mode

# Or use the alias
bronchmonkey run --edition lite
```

### 5. Docker Migration

#### Old Docker Commands
```bash
# Different compose files per branch
docker-compose up                    # main branch
docker-compose -f lite.yml up       # lite branch
docker build -f Dockerfile.space .  # space
```

#### New Docker Commands
```bash
# Unified compose files
docker-compose -f compose/docker-compose.full.yml up
docker-compose -f compose/docker-compose.lite.yml up
docker build -f docker/Dockerfile.space -t bronchmonkey:space .
```

### 6. API Changes

#### Search API
```python
# Old (direct import)
from indexing.hybrid_search import HybridSearcher
searcher = HybridSearcher(faiss_path, bm25_path)

# New (config-based)
from ipchat.core.config import load_config
from ipchat.core.retrieval import create_hybrid_searcher
config = load_config(edition="lite")
searcher = create_hybrid_searcher(config)
```

#### Citation Formatting
```python
# Old
from utils.citations import format_citations
citations = format_citations(sources)

# New
from ipchat.core.citations import CitationFormatter
formatter = CitationFormatter(style="mla")
citations = formatter.format(sources)
```

### 7. Configuration Management

The new system uses Pydantic-based configuration:

```python
# config.json (optional)
{
  "edition": "lite",
  "llm": {
    "model": "gpt-4o-mini",
    "temperature": 0.3
  },
  "retrieval": {
    "num_results": 10,
    "vector_weight": 0.5
  }
}
```

Load with:
```python
from ipchat.core.config import load_config
config = load_config(config_file="config.json")
```

## Edition Feature Matrix

| Feature | Full | Lite | Space |
|---------|------|------|-------|
| Streamlit UI | ✅ | ✅ | ✅ |
| FastAPI Server | ✅ | ❌ | ❌ |
| PostgreSQL | ✅ | ❌ | ❌ |
| FAISS Index | ✅ | ✅ | ✅ |
| BM25 Search | ✅ | ✅ | ✅ |
| Depth Mode | ✅ | ✅ | ✅ |
| Auth Support | ✅ | Optional | ✅ |
| Docker Support | ✅ | ✅ | ✅ |

## Testing Your Migration

Run the verification command to ensure everything is set up correctly:

```bash
ipchat verify
```

This will check:
- Python version compatibility
- Required packages installed
- Data files present
- Environment variables set
- Database connectivity (full edition)

## Rollback Plan

If you need to rollback:

```bash
# Switch back to old branch
git checkout main  # or lite-perf

# Reinstall old dependencies
pip install -r requirements.txt
```

## Common Issues and Solutions

### Issue: Module not found errors
**Solution**: Ensure you've installed the package with the correct extras:
```bash
pip install -e ".[full]"  # includes all dependencies
```

### Issue: Data files not found
**Solution**: The new structure expects data in `./data/`. Copy or symlink your existing data:
```bash
ln -s /path/to/old/data ./data
```

### Issue: Port conflicts
**Solution**: The new system uses standard ports by default. Override with:
```bash
ipchat run --edition lite --port 8502
```

### Issue: Database connection errors (full edition)
**Solution**: Update your connection string:
```bash
export IPCHAT_INFRA__SQL_CONNECTION_STRING="postgresql://user:pass@localhost/dbname"
```

## Getting Help

- Check the CLI help: `ipchat --help`
- View current config: `ipchat info`
- Enable debug mode: `ipchat --debug run --edition lite`

## Benefits of the New Architecture

1. **Single Codebase**: No more branch switching or code duplication
2. **Unified CLI**: One command to rule them all
3. **Flexible Configuration**: Easy to customize per deployment
4. **Better Testing**: Test all editions from one place
5. **Cleaner Dependencies**: Install only what you need
6. **Easier Maintenance**: Update once, benefit everywhere

---

For questions or issues, please open an issue on GitHub.