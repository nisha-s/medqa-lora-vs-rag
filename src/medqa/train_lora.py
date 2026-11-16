"""LoRA fine-tuning with PEFT. Supports resuming from the last checkpoint.

    python -m medqa.train_lora --config configs/base.yaml --seed 42
"""

from pathlib import Path
from typing import Any


def build_model(model_name: str, lora_cfg: dict[str, Any], device: str):
    """Load the base model and wrap it with a LoRA adapter."""
    raise NotImplementedError


def train(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fine-tune, save the adapter and return training stats (time, peak memory)."""
    raise NotImplementedError


def adapter_dir(output_dir: str | Path, model_name: str, seed: int) -> Path:
    """Where the adapter for a given model and seed is stored."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented yet")
