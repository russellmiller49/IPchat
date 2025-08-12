#!/usr/bin/env bash
set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(pwd)}"

cd "$REPO_ROOT"

echo "==> Ensuring we're on lite-perf branch"
git checkout lite-perf || git checkout -b lite-perf

echo "==> Copying files into repo"
rsync -av "$PATCH_DIR/" "$REPO_ROOT/" --exclude "apply_gpt5_update.sh"

echo "==> Staging and committing"
git add utils/openai_client.py chatbot_app.py requirements.openai.txt Dockerfile docker-compose.lite.yml deployment/HF_SPACE_README.md
git commit -m "GPT-5 update: model toggle, robust OpenAI client, and minimal OpenAI SDK pin"

echo "==> Done."
echo "Next:"
echo "  docker compose -f docker-compose.lite.yml build --no-cache"
echo "  docker compose -f docker-compose.lite.yml up"
