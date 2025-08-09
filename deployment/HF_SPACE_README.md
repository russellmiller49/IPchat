---
title: Bronchmonkey (Lite, GPT‑5 Ready)
sdk: docker
app_port: 8501
---

# Deploy to Hugging Face Spaces (Docker)

1) Create a **Docker Space** on HF and push this branch.
2) Add **Variables/Secrets**:
   - `OPENAI_API_KEY` (required)
   - Optional: `GEN_MODEL` default (e.g., `gpt-5-mini`)
   - Optional: `EMBED_MODEL` (defaults to `intfloat/e5-small-v2`)
   - Optional: `HF_HOME=/data/.huggingface` (enable Persistent Storage so cache survives restarts)
3) Build and run—Streamlit on port `8501`; FastAPI runs privately on `localhost:8000` in the same container.
4) Use the **sidebar** to switch between **Fast (gpt-5-mini)** and **Max (gpt-5)** at runtime.
