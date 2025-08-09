# GPT‑5 Update (lite‑perf)

This update bundles:
- A robust OpenAI client wrapper that supports both the new and legacy SDKs.
- A sidebar quality toggle to pick **gpt-5-mini** (fast/cheap) or **gpt-5** (max quality).
- A minimal OpenAI SDK pin installed separately from your main requirements.

## Apply
```bash
# from your repo root on the lite-perf branch
unzip ~/Downloads/ipchat-gpt5-update.zip -d /tmp/ipchat-gpt5-update
bash /tmp/ipchat-gpt5-update/apply_gpt5_update.sh $(pwd)

docker compose -f docker-compose.lite.yml build --no-cache
docker compose -f docker-compose.lite.yml up
```

## Notes
- The UI shows the active model in the sidebar.
- If you set `GEN_MODEL` in the environment, it will appear as a third option ("Env (...)").
- No need to apply the previous “hotfix”—this update **includes it**.

## Rollback
```bash
git reset --hard HEAD~1
```
