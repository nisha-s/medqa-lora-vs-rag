"""Tokenisation and label masking for LoRA training."""

import pandas as pd

from medqa.train_lora import IGNORE_INDEX, adapter_dir, tokenize_example, tokenize_split


class WordTokenizer:
    """Fake tokenizer: one id per whitespace token, eos id 0."""

    eos_token_id = 0

    def __call__(self, text: str) -> dict:
        return {"input_ids": [hash(w) % 1000 + 1 for w in text.split()]}


def test_prompt_is_masked_and_eos_appended() -> None:
    tok = WordTokenizer()
    ex = tokenize_example(
        tok,
        "### Symptoms: cough\n### Disease:",
        " flu",
        max_len=128,
        loss_on_prompt=False,
    )
    prompt_len = 5  # "###", "Symptoms:", "cough", "###", "Disease:"
    assert ex["input_ids"][-1] == tok.eos_token_id
    assert ex["labels"][:prompt_len] == [IGNORE_INDEX] * prompt_len
    assert ex["labels"][prompt_len:] == ex["input_ids"][prompt_len:]
    assert ex["attention_mask"] == [1] * len(ex["input_ids"])


def test_loss_on_prompt_keeps_all_labels() -> None:
    ex = tokenize_example(
        WordTokenizer(), "a b c", " d", max_len=128, loss_on_prompt=True
    )
    assert ex["labels"] == ex["input_ids"]


def test_truncation_is_counted(caplog) -> None:
    frame = pd.DataFrame(
        {"symptom_text": ["one two three four five"], "disease": ["flu"]}
    )
    with caplog.at_level("WARNING"):
        examples = tokenize_split(
            WordTokenizer(), frame, max_len=4, loss_on_prompt=False
        )
    assert len(examples[0]["input_ids"]) == 4
    assert "lost their answer" in caplog.text


def test_adapter_dir_is_model_and_seed_specific() -> None:
    path = adapter_dir("outputs", "TinyLlama/TinyLlama-1.1B", 42)
    assert str(path) == "outputs/adapters/TinyLlama__TinyLlama-1.1B_seed42"
