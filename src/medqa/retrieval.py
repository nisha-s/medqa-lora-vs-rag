"""Retrievers over training cases: MiniLM embeddings (cosine) and Jaccard.

Both retrievers share one interface: build(train) stores the training cases,
query_batch(queries, k) returns an (n_queries, k) array of training indices,
nearest first.

Run as a module to build and cache the embedding index:
    python -m medqa.retrieval --config configs/base.yaml --build-index
"""

import argparse
import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medqa.config import load_config
from medqa.data import load_splits

log = logging.getLogger(__name__)


class EmbeddingRetriever:
    """Cosine similarity over sentence-transformer embeddings, cached to disk."""

    def __init__(
        self, model_name: str, cache_dir: str | Path, device: str | None = None
    ) -> None:
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.device = device  # None means pick automatically when the model loads
        self._model = None
        self.embeddings: np.ndarray | None = None
        self.labels: list[str] = []

    def _load_model(self):
        """Load the sentence-transformer lazily so cached runs skip the import."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            from medqa.config import pick_device

            self._model = SentenceTransformer(
                self.model_name, device=self.device or pick_device()
            )
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Unit-normalised embeddings, so cosine similarity is a dot product."""
        model = self._load_model()
        return model.encode(
            texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

    def _cache_path(self, texts: list[str]) -> Path:
        digest = hashlib.sha1("\n".join([self.model_name, *texts]).encode()).hexdigest()
        slug = self.model_name.replace("/", "__")
        return self.cache_dir / f"{slug}_{digest[:12]}.npy"

    def build(self, train: pd.DataFrame) -> None:
        """Embed every training case, reading from the cache when present."""
        texts = train["symptom_text"].tolist()
        self.labels = train["disease"].tolist()
        path = self._cache_path(texts)
        if path.exists():
            self.embeddings = np.load(path)
            log.info("loaded cached embeddings from %s", path)
            return
        log.info("embedding %d training cases", len(texts))
        self.embeddings = self._encode(texts)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, self.embeddings)

    def query_batch(self, queries: pd.DataFrame, k: int) -> np.ndarray:
        """Indices of the k most similar training cases for each query row."""
        if self.embeddings is None:
            raise RuntimeError("call build() before query_batch()")
        query_embeddings = self._encode(queries["symptom_text"].tolist())
        similarity = query_embeddings @ self.embeddings.T
        return top_k_indices(similarity, k)


class JaccardRetriever:
    """Jaccard similarity over symptom sets, computed with a binary matrix."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.matrix: np.ndarray | None = None
        self.labels: list[str] = []

    def _vectorize(
        self, symptom_lists: list[tuple[str, ...]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Binary matrix over the training vocabulary plus each row's true set size.

        Symptoms outside the vocabulary cannot match anything, but they still
        count towards the set size so the union is right.
        """
        matrix = np.zeros((len(symptom_lists), len(self.vocab)), dtype=np.float32)
        sizes = np.zeros(len(symptom_lists), dtype=np.float32)
        for i, symptoms in enumerate(symptom_lists):
            sizes[i] = len(symptoms)
            for symptom in symptoms:
                j = self.vocab.get(symptom)
                if j is not None:
                    matrix[i, j] = 1.0
        return matrix, sizes

    def build(self, train: pd.DataFrame) -> None:
        """Store the training symptom sets as a binary matrix."""
        self.labels = train["disease"].tolist()
        symptom_lists = list(train["symptoms"])
        self.vocab = {
            s: i
            for i, s in enumerate(sorted({s for row in symptom_lists for s in row}))
        }
        self.matrix, self._train_sizes = self._vectorize(symptom_lists)

    def query_batch(self, queries: pd.DataFrame, k: int) -> np.ndarray:
        """Indices of the k training cases with the highest Jaccard overlap."""
        if self.matrix is None:
            raise RuntimeError("call build() before query_batch()")
        query_matrix, query_sizes = self._vectorize(list(queries["symptoms"]))
        intersection = query_matrix @ self.matrix.T
        union = query_sizes[:, None] + self._train_sizes[None, :] - intersection
        similarity = np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0)
        return top_k_indices(similarity, k)


def top_k_indices(similarity: np.ndarray, k: int) -> np.ndarray:
    """Row-wise indices of the k largest values, sorted most similar first."""
    k = min(k, similarity.shape[1])
    # argpartition is O(n); the final sort only touches k items per row.
    candidates = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, candidates, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def build_retriever(
    config: dict[str, Any], name: str | None = None, device: str | None = None
):
    """Construct the retriever named in config (or `name` if given), unbuilt.

    Torch is only imported for the embedding retriever, so Jaccard runs stay light.
    """
    cfg = config["retrieval"]
    name = name or cfg["retriever"]
    if name == "embedding":
        return EmbeddingRetriever(cfg["embedding_model"], cfg["cache_dir"], device)
    if name == "jaccard":
        return JaccardRetriever()
    raise ValueError(f"unknown retriever {name!r}; expected 'embedding' or 'jaccard'")


def majority_label(labels: list[str]) -> str:
    """Majority vote; ties go to the label that appears earliest (nearest case)."""
    if not labels:
        raise ValueError("no labels to vote on")
    counts = Counter(labels)
    best = max(counts.values())
    for label in labels:
        if counts[label] == best:
            return label
    raise AssertionError("unreachable")


def recall_at_k(retrieved_labels: list[str], true_label: str) -> bool:
    """Whether the true disease is among the retrieved cases' labels."""
    return true_label in retrieved_labels


def retrieved_cases(train: pd.DataFrame, indices: np.ndarray) -> list[tuple[str, str]]:
    """(symptom_text, disease) pairs for one query's neighbours, nearest first."""
    rows = train.iloc[indices]
    return list(zip(rows["symptom_text"], rows["disease"]))


def main(config: dict[str, Any], build_index: bool) -> None:
    """Build the embedding index from the saved train split."""
    train = load_splits(config["data"]["splits_dir"], names=("train",))["train"]
    if build_index:
        retriever = build_retriever(config, name="embedding")
        retriever.build(train)
        log.info("index ready: %s", retriever.embeddings.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--build-index", action="store_true")
    args = parser.parse_args()
    main(load_config(args.config), args.build_index)
