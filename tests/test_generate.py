"""Generation helpers and prompt assembly."""

from medqa.evaluate import run_name
from medqa.generate import first_line
from medqa.prompts import rag_prompt, training_example, zero_shot_prompt


def test_first_line_cuts_at_newline() -> None:
    assert first_line(" panic disorder\n### Symptoms: cough") == "panic disorder"
    # Leading blank lines are stripped first, as in the notebook.
    assert first_line("\n\nflu") == "flu"


def test_zero_shot_prompt_matches_notebook_format() -> None:
    assert (
        zero_shot_prompt("cough, fever") == "### Symptoms: cough, fever\n### Disease:"
    )


def test_rag_prompt_puts_examples_before_query() -> None:
    prompt = rag_prompt([("cough", "flu"), ("rash", "measles")], "cough, rash")
    assert prompt == (
        "### Symptoms: cough\n### Disease: flu\n\n"
        "### Symptoms: rash\n### Disease: measles\n\n"
        "### Symptoms: cough, rash\n### Disease:"
    )


def test_training_example_splits_prompt_and_completion() -> None:
    prompt, completion = training_example("cough", "flu")
    assert prompt + completion == "### Symptoms: cough\n### Disease: flu"


def test_run_name() -> None:
    assert run_name("base_zero_shot", None, None, None) == "base_zero_shot"
    assert run_name("base_rag", "jaccard", 5, None) == "base_rag_jaccard_k5"
    assert run_name("lora_rag", "jaccard", 5, 42) == "lora_rag_jaccard_k5_seed42"
