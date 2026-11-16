"""Greedy generation with short outputs, cut at the first newline."""

from typing import Any


def load_model(model_name: str, adapter_path: str | None, device: str):
    """Load base weights, optionally with a LoRA adapter merged in."""
    raise NotImplementedError


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy-decode a single prompt and return the first line of the completion."""
    raise NotImplementedError


def generate_batch(
    model, tokenizer, prompts: list[str], max_new_tokens: int, batch_size: int
) -> list[dict[str, Any]]:
    """Generate for many prompts, recording latency and prompt length per example."""
    raise NotImplementedError
