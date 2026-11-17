"""Symptom-to-text conversion and prompt templates.

The prompt format is carried over from lora-medical-qa:
    ### Symptoms: <comma-separated symptoms>
    ### Disease: <disease>
"""

SYMPTOMS_HEADER = "### Symptoms:"
DISEASE_HEADER = "### Disease:"


def symptoms_to_text(symptoms: tuple[str, ...]) -> str:
    """Join symptom names with commas, matching the original notebook."""
    return ", ".join(symptoms) if symptoms else "No symptoms"


def zero_shot_prompt(symptom_text: str) -> str:
    """Prompt with just the query case; the model continues after the header."""
    return f"{SYMPTOMS_HEADER} {symptom_text}\n{DISEASE_HEADER}"


def format_case(symptom_text: str, disease: str) -> str:
    """One complete case, used both for RAG examples and for training."""
    return f"{zero_shot_prompt(symptom_text)} {disease}"


def rag_prompt(retrieved: list[tuple[str, str]], symptom_text: str) -> str:
    """Top-k (symptom_text, disease) examples followed by the query."""
    examples = [format_case(text, disease) for text, disease in retrieved]
    return "\n\n".join(examples + [zero_shot_prompt(symptom_text)])


def training_example(symptom_text: str, disease: str) -> tuple[str, str]:
    """Return (prompt, completion) so training can mask the loss on the prompt.

    The end-of-sequence token is appended by the tokenizer at training time.
    """
    return zero_shot_prompt(symptom_text), f" {disease}"
