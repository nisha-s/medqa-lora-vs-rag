#!/usr/bin/env bash
# Run every experiment arm for every seed, then aggregate results.
# Usage: bash scripts/run_all.sh configs/base.yaml
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:src"

# Arms that need no training.
python -m medqa.evaluate --config "$CONFIG" --arm retrieval_only
python -m medqa.evaluate --config "$CONFIG" --arm base_zero_shot
python -m medqa.evaluate --config "$CONFIG" --arm base_rag

# LoRA arms, one adapter per seed.
for SEED in $(python -c "import yaml,sys; print(*yaml.safe_load(open('$CONFIG'))['seeds'])" 2>/dev/null || echo 42); do
  python -m medqa.train_lora --config "$CONFIG" --seed "$SEED"
  python -m medqa.evaluate --config "$CONFIG" --arm lora --seed "$SEED"
  python -m medqa.evaluate --config "$CONFIG" --arm lora_rag --seed "$SEED"
done

# General-knowledge retention for base and LoRA weights.
python -m medqa.evaluate --config "$CONFIG" --mmlu

python -m medqa.evaluate --config "$CONFIG" --aggregate
