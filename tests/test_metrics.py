"""Metric tests: normalisation, exact match, ROUGE-L and majority vote."""

import numpy as np
import pandas as pd

from medqa import metrics
from medqa.retrieval import JaccardRetriever, majority_label, recall_at_k, top_k_indices


def test_normalize_strips_case_punctuation_whitespace() -> None:
    assert metrics.normalize("  Panic   Disorder! ") == "panic disorder"
    assert metrics.normalize("Crohn's disease") == "crohn s disease"


def test_exact_match_ignores_formatting() -> None:
    assert metrics.exact_match("Panic disorder.", "panic disorder")
    assert not metrics.exact_match("panic", "panic disorder")


def test_rouge_l_matches_notebook_behaviour() -> None:
    assert metrics.rouge_l("panic disorder", "panic disorder") == 1.0
    assert metrics.rouge_l("", "panic disorder") == 0.0
    assert metrics.rouge_l("flu", "panic disorder") == 0.0
    # LCS = 1 of hyp len 2 and ref len 2 -> precision = recall = 0.5 -> F1 = 0.5
    assert metrics.rouge_l("panic attack", "panic disorder") == 0.5


def test_majority_label_tie_goes_to_nearest() -> None:
    assert majority_label(["a", "b", "b", "a"]) == "a"
    assert majority_label(["b", "a", "a"]) == "a"
    assert recall_at_k(["a", "b"], "b") and not recall_at_k(["a"], "b")


def test_top_k_indices_sorted_most_similar_first() -> None:
    sim = np.array([[0.1, 0.9, 0.5, 0.7]])
    assert top_k_indices(sim, 3).tolist() == [[1, 3, 2]]
    assert top_k_indices(sim, 10).shape == (1, 4)


def test_jaccard_retriever_finds_exact_set_first() -> None:
    train = pd.DataFrame(
        {
            "symptoms": [("a", "b"), ("a", "b", "c"), ("d",)],
            "symptom_text": ["a, b", "a, b, c", "d"],
            "disease": ["x", "y", "z"],
        }
    )
    retriever = JaccardRetriever()
    retriever.build(train)
    queries = pd.DataFrame(
        {"symptoms": [("a", "b", "c"), ("d", "unknown")], "symptom_text": ["", ""]}
    )
    neighbours = retriever.query_batch(queries, 2)
    assert neighbours[0].tolist() == [1, 0]
    assert neighbours[1][0] == 2


def test_mmlu_prompt_lists_four_choices() -> None:
    prompt = metrics.mmlu_prompt(
        "What is 2+2?", ["3", "4", "5", "6"], "high_school_mathematics"
    )
    assert prompt.startswith(
        "The following is a multiple choice question about high school mathematics."
    )
    assert "\nB. 4\n" in prompt
    assert prompt.endswith("Answer:")


def test_answer_token_ids_uses_last_token() -> None:
    class Tok:
        def encode(self, text, add_special_tokens):
            return [99, ord(text.strip())]

    assert metrics.answer_token_ids(Tok()) == [65, 66, 67, 68]
