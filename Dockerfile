# Optimized Dockerfile for a small, fast deploy
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HF_HOME=/app/.cache/huggingface

WORKDIR /app

# System deps: keep it lean. libgomp1 is needed by faiss-cpu wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-pull smaller embedding model so first run is snappy
# (This warms the HF cache inside the image; on HF Spaces, also set HF_HOME to a persistent volume)
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/e5-small-v2")
PY

# App code
COPY . .

# Expose ports (FastAPI + Streamlit)
EXPOSE 8000 8501

# Basic healthcheck against the FastAPI docs endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

# Run both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 & streamlit run chatbot_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
