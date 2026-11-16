"""Retrievers over training cases: MiniLM embeddings (cosine) and Jaccard.

Run as a module to build and cache the embedding index:
    python -m medqa.retrieval --config configs/base.yaml --build-index
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class EmbeddingRetriever:
    """Cosine similarity over sentence-transformer embeddings, cached to disk."""

    def __init__(self, model_name: str, cache_dir: str | Path, device: str) -> None:
        raise NotImplementedError

    def build(self, cases: pd.DataFrame) -> None:
        """Embed every training case and cache the matrix."""
        raise NotImplementedError

    def query(self, symptom_text: str, k: int) -> list[int]:
        """Return indices of the k most similar training cases."""
        raise NotImplementedError


class JaccardRetriever:
    """Jaccard similarity over symptom sets. No model needed."""

    def __init__(self) -> None:
        raise NotImplementedError

    def build(self, cases: pd.DataFrame) -> None:
        """Store the training symptom sets."""
        raise NotImplementedError

    def query(self, symptoms: tuple[str, ...], k: int) -> list[int]:
        """Return indices of the k training cases with highest Jaccard overlap."""
        raise NotImplementedError


def majority_label(labels: list[str]) -> str:
    """Majority vote for the retrieval-only baseline; ties go to the nearest case."""
    raise NotImplementedError


def recall_at_k(retrieved_labels: list[str], true_label: str) -> bool:
    """Whether the true disease is among the retrieved cases' labels."""
    raise NotImplementedError


def main(config: dict[str, Any]) -> None:
    """Build the retrieval index from the saved train split."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented yet")
