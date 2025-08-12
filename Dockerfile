# Minimal Dockerfile patch: install openai pin separately so we don't disturb your existing pins.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY requirements.openai.txt .
RUN pip install --no-cache-dir -r requirements.openai.txt

# Pre-pull a small embedding model cache for fast cold-starts
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/e5-small-v2")
PY

COPY . .

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/docs || exit 1

CMD ["sh", "-c", "uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 & streamlit run chatbot_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
