# Lite-Perf Branch — Small, Fast, and Cheap

This branch focuses on **low latency** and **low cost** for a small group (<10 users) without losing answer quality.

## Highlights
- **Small embeddings**: `intfloat/e5-small-v2` for fast indexing & query.
- **Answer caching**: Avoids re-paying for identical questions.
- **Cheaper gen model**: Default `GEN_MODEL=gpt-4o-mini` (configurable).
- **Docker optimized**: slimmer image, pre-pulls embedding model cache.
- **Lite compose**: run **without Postgres** (hybrid search still works; SQL bits gracefully skip).
- **HF Spaces ready**: one public port (Streamlit), internal FastAPI stays on localhost.

## Quick Start (Local Lite)
```bash
docker compose -f docker-compose.lite.yml up --build
# open http://localhost:8501
```

## Hugging Face Spaces (Docker)
See `deployment/HF_SPACE_README.md` for step-by-step instructions.

## What changed
- `backend/api/main.py`: `EMBED_MODEL` now env-driven; default `intfloat/e5-small-v2`.
- `indexing/build_faiss.py`: default model small; env-aware.
- `chunking/chunker.py`: smaller chunks (size=450 overlap=80), env-overridable.
- `chatbot_app.py`: cached answers; configurable `GEN_MODEL`; request timeouts.
- `Dockerfile`: slimmer; pre-downloads the small embedding model; sets `HF_HOME` for cache.
- `docker-compose.lite.yml`: single-container, no Postgres.
- `utils/page_images.py`: helper to render cited PDF pages (optional nice-to-have).
