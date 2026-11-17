#!/usr/bin/env bash
# Run every experiment arm for every seed, then aggregate results.
# Usage: bash scripts/run_all.sh configs/base.yaml
set -euo pipefail

CONFIG="${1:-configs/base.yaml}"
export PYTHONPATH="${PYTHONPATH:-}:src"

read_list() { python -c "import yaml,sys; v=yaml.safe_load(open('$CONFIG'))$1; print(*v)"; }
SEEDS=$(read_list "['seeds']")
KS=$(read_list "['retrieval']['k_ablation']")

# Retrieval-only control: both retrievers, every k. No LLM, cheap.
for RETRIEVER in jaccard embedding; do
  for K in $KS; do
    python -m medqa.evaluate --config "$CONFIG" --arm retrieval_only --retriever "$RETRIEVER" --k "$K"
  done
done

# Base model arms, no training.
python -m medqa.evaluate --config "$CONFIG" --arm base_zero_shot
for K in $KS; do
  python -m medqa.evaluate --config "$CONFIG" --arm base_rag --k "$K"
done

# LoRA arms, one adapter per seed.
for SEED in $SEEDS; do
  python -m medqa.train_lora --config "$CONFIG" --seed "$SEED"
  python -m medqa.evaluate --config "$CONFIG" --arm lora --seed "$SEED"
  python -m medqa.evaluate --config "$CONFIG" --arm lora_rag --seed "$SEED"
done

# General-knowledge retention for base and LoRA weights.
python -m medqa.evaluate --config "$CONFIG" --mmlu

python -m medqa.evaluate --config "$CONFIG" --aggregate
