---
title: Bronchmonkey (Lite)
sdk: docker
app_port: 8501
---

# Deploy to Hugging Face Spaces (Docker)

1) Create a **Docker Space** and push this repo/branch to it.
2) Add **Secrets** (Settings → Variables):
   - `OPENAI_API_KEY`: your key
   - (optional) `GEN_MODEL`: defaults to `gpt-4o-mini`
   - (optional) `EMBED_MODEL`: defaults to `intfloat/e5-small-v2`
   - (optional) `DATABASE_URL`: if you plan to use the SQL features
3) Enable **Persistent Storage** and set an environment variable:
   - `HF_HOME=/data/.huggingface`  (so models stay cached across restarts)
4) The Space exposes only one public port (Streamlit on `8501`). The FastAPI
   server runs privately on `localhost:8000` inside the container.
5) Build and run — you should see the Streamlit UI.

Notes:
- For fastest cold-starts, we pre-download the embedding model in the Dockerfile.
- If you later switch to a different embedding model, rebuild the Space.
