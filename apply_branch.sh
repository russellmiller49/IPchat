#!/usr/bin/env bash
set -euo pipefail

BRANCH_NAME="lite-perf"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo ".")"

cd "$REPO_ROOT"

echo "==> Creating branch: ${BRANCH_NAME}"
git checkout -b "${BRANCH_NAME}" || git checkout "${BRANCH_NAME}"

echo "==> Copying optimized files into repo"
rsync -av --exclude 'apply_branch.sh' "$PATCH_DIR/" "$REPO_ROOT/"

echo "==> Staging changes"
git add Dockerfile docker-compose.lite.yml backend/api/main.py indexing/build_faiss.py chunking/chunker.py chatbot_app.py utils/page_images.py deployment/HF_SPACE_README.md deployment/FLYIO.md README-LITE.md || true

echo "==> Committing"
git commit -m "lite-perf: faster small-model embeddings, cached answers, HF Space-compatible Dockerfile, and lite compose"

echo "==> Done."
echo ""
echo "Next steps:"
echo "  - For Hugging Face Spaces (Docker): open deployment/HF_SPACE_README.md"
echo "  - For local or VM: docker compose -f docker-compose.lite.yml up --build"
