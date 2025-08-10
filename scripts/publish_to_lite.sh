#!/usr/bin/env bash
set -euo pipefail
LITE_REPO="${1:-../IPchat}"  # path to a lite-perf checkout
if [ ! -d "$LITE_REPO" ]; then
  echo "Usage: $0 /path/to/IPchat (lite-perf checkout)"; exit 1;
fi
mkdir -p "$LITE_REPO/data/index" "$LITE_REPO/data/chunks"
rsync -av data/index/ "$LITE_REPO/data/index/"
rsync -av data/chunks/chunks.jsonl "$LITE_REPO/data/chunks/chunks.jsonl"
echo ">> Copied artifacts to $LITE_REPO"