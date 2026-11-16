"""Symptom-to-text conversion and prompt templates.

Prompt format is carried over from lora-medical-qa:
    ### Symptoms: <comma-separated symptoms>
    ### Disease:
"""


def symptoms_to_text(symptoms: tuple[str, ...]) -> str:
    """Convert a symptom set to natural text, matching the original notebook."""
    raise NotImplementedError


def zero_shot_prompt(symptom_text: str) -> str:
    """Prompt with just the query case."""
    raise NotImplementedError


def rag_prompt(retrieved: list[tuple[str, str]], symptom_text: str) -> str:
    """Prompt with top-k (symptom_text, disease) examples followed by the query."""
    raise NotImplementedError


def training_example(symptom_text: str, disease: str) -> str:
    """Prompt plus answer, used for LoRA fine-tuning."""
    raise NotImplementedError
