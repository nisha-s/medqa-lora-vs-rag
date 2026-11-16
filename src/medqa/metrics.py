"""Scoring: exact match, semantic similarity, ROUGE-L and MMLU log-likelihood."""


def normalize(text: str) -> str:
    """Lowercase and strip punctuation and whitespace."""
    raise NotImplementedError


def exact_match(prediction: str, target: str) -> bool:
    """Exact match after normalisation. Primary metric."""
    raise NotImplementedError


def semantic_similarity(prediction: str, target: str, embedder) -> float:
    """Cosine similarity between embeddings, carried over from lora-medical-qa."""
    raise NotImplementedError


def rouge_l(prediction: str, target: str) -> float:
    """ROUGE-L F1, carried over from lora-medical-qa."""
    raise NotImplementedError


def mmlu_accuracy(model, tokenizer, questions: list[dict], device: str) -> float:
    """Score each question by log-likelihood of the answer letters A-D."""
    raise NotImplementedError
