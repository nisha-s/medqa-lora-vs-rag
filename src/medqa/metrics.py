"""Scoring helpers: exact match, semantic similarity, ROUGE-L, and MMLU log-likelihood."""

import re
import string

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = _PUNCT.sub(" ", str(text).lower())
    return _SPACES.sub(" ", text).strip()


def exact_match(prediction: str, target: str) -> bool:
    """Exact match after normalisation. Primary metric."""
    return normalize(prediction) == normalize(target)


def rouge_l(prediction: str, target: str) -> float:
    """ROUGE-L F1 over whitespace tokens, carried over from lora-medical-qa."""
    hyp = str(prediction).lower().split()
    ref = str(target).lower().split()
    if not hyp or not ref:
        return 0.0
    # Longest common subsequence, filled one DP row at a time to keep memory flat.
    previous = [0] * (len(ref) + 1)
    for h in hyp:
        current = [0]
        for j, r in enumerate(ref, start=1):
            if h == r:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[j - 1]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def semantic_similarity(
    predictions: list[str], targets: list[str], embedder
) -> list[float]:
    """Cosine similarity between MiniLM embeddings, clipped at 0 like the notebook did.

    Encoded in one batch; doing it a pair at a time was the notebook's main
    bottleneck.
    """
    import numpy as np

    pred = embedder.encode(
        list(predictions), normalize_embeddings=True, convert_to_numpy=True
    )
    targ = embedder.encode(
        list(targets), normalize_embeddings=True, convert_to_numpy=True
    )
    scores = np.einsum("ij,ij->i", pred, targ)
    return [float(max(0.0, s)) for s in scores]


MMLU_LETTERS = ("A", "B", "C", "D")


def mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    """Standard zero-shot MMLU prompt; the model continues with a letter."""
    lines = [
        f"The following is a multiple choice question about {subject.replace('_', ' ')}.",
        "",
        question.strip(),
    ]
    for letter, choice in zip(MMLU_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def answer_token_ids(tokenizer) -> list[int]:
    """Token id of each answer letter as it would follow 'Answer:'."""
    ids = []
    for letter in MMLU_LETTERS:
        encoded = tokenizer.encode(f" {letter}", add_special_tokens=False)
        ids.append(encoded[-1])
    return ids


def mmlu_accuracy(
    model, tokenizer, questions: list[dict], batch_size: int
) -> tuple[float, list[dict]]:
    """Score each question by the log-likelihood of the four answer letters.

    `questions` rows need subject, question, choices (list of 4) and answer (0-3).
    Returns overall accuracy and one row per question with the four log-probs.
    """
    import torch

    device = next(model.parameters()).device
    letter_ids = torch.tensor(answer_token_ids(tokenizer), device=device)
    tokenizer.padding_side = "left"
    rows: list[dict] = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        prompts = [
            mmlu_prompt(q["question"], q["choices"], q["subject"]) for q in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits[:, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)[:, letter_ids].cpu()
        predicted = log_probs.argmax(dim=-1).tolist()
        for q, pred, lp in zip(batch, predicted, log_probs.tolist()):
            rows.append(
                {
                    "subject": q["subject"],
                    "question": q["question"],
                    "answer": MMLU_LETTERS[q["answer"]],
                    "predicted": MMLU_LETTERS[pred],
                    "correct": pred == q["answer"],
                    **{
                        f"logprob_{letter}": value
                        for letter, value in zip(MMLU_LETTERS, lp)
                    },
                }
            )
    accuracy = sum(r["correct"] for r in rows) / len(rows) if rows else 0.0
    return accuracy, rows
