"""Metric tests: normalisation, exact match, ROUGE-L and majority vote."""

import pytest


@pytest.mark.skip(reason="metrics.py not implemented yet")
def test_normalize_strips_case_punctuation_whitespace() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="metrics.py not implemented yet")
def test_exact_match_ignores_formatting() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="metrics.py not implemented yet")
def test_rouge_l_identical_strings_is_one() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="retrieval.py not implemented yet")
def test_majority_label_tie_goes_to_nearest() -> None:
    raise NotImplementedError
