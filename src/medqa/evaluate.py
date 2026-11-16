"""Run an experiment arm, save per-example predictions and aggregate results.

    python -m medqa.evaluate --config configs/base.yaml --arm base_rag
    python -m medqa.evaluate --config configs/base.yaml --mmlu
    python -m medqa.evaluate --config configs/base.yaml --aggregate
"""

from typing import Any

import pandas as pd

ARMS = ("base_zero_shot", "base_rag", "lora", "lora_rag", "retrieval_only")


def run_arm(config: dict[str, Any], arm: str, seed: int) -> pd.DataFrame:
    """Evaluate one arm on the test split and return per-example rows."""
    raise NotImplementedError


def run_mmlu(config: dict[str, Any]) -> pd.DataFrame:
    """MMLU retention for the base model and each LoRA adapter."""
    raise NotImplementedError


def aggregate(config: dict[str, Any]) -> pd.DataFrame:
    """Combine per-run CSVs into mean and std across seeds."""
    raise NotImplementedError


def log_to_wandb(config: dict[str, Any], run_name: str, metrics: dict, samples: pd.DataFrame) -> None:
    """Log metrics and a sample table if wandb.enabled is true; no-op otherwise."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented yet")
