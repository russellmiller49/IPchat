# Tiny VM Alternative (Fly.io)

If you prefer a small VM:
1) Install flyctl and run `fly launch` in the repo.
2) Set secrets:
   ```bash
   fly secrets set OPENAI_API_KEY=...
   fly secrets set GEN_MODEL=gpt-4o-mini
   fly secrets set EMBED_MODEL=intfloat/e5-small-v2
   # (optional) DATABASE_URL=postgresql://user:pass@host/db
   ```
3) Expose only port 8501 externally in `fly.toml`:
   ```toml
   [[services]]
     internal_port = 8501
     protocol = "tcp"
     [[services.ports]]
       port = 80
     [[services.ports]]
       port = 443
   ```
   The app still runs FastAPI on 8000 internally.
4) `fly deploy`

Tip: start with a shared-cpu 256–512MB instance; bump if you see OOM kills.
