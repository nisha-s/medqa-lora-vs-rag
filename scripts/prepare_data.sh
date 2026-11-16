#!/usr/bin/env bash
# Build deduplicated, leakage-free splits and cache retrieval embeddings.
# Usage: bash scripts/prepare_data.sh configs/base.yaml
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:src"

python -m medqa.data --config "$CONFIG"
python -m medqa.retrieval --config "$CONFIG" --build-index
