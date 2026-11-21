"""Greedy decoding. Outputs are short and everything past the first newline is dropped."""

import time
from typing import Any


def first_line(text: str) -> str:
    """The model's answer is everything up to the first newline."""
    return text.strip().split("\n")[0].strip()


def pick_dtype(device: str):
    """bfloat16 on CUDA, which is what TinyLlama was trained in; float32 otherwise."""
    import torch

    return torch.bfloat16 if device == "cuda" else torch.float32


def load_model(model_name: str, adapter_path: str | None, device: str):
    """Load base weights, optionally with a LoRA adapter merged in for inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Pad and truncate on the left. Padding left lines up every prompt's last
    # token at the same position in the batch; truncating left means an
    # over-long RAG prompt loses its oldest examples, never the query itself.
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=pick_dtype(device))
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, tokenizer


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    batch_size: int,
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    """Greedy-decode every prompt, recording latency and prompt length per example."""
    import torch

    device = next(model.parameters()).device
    results: list[dict[str, Any]] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(device)

        began = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - began) * 1000 / len(batch)

        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        prompt_tokens = encoded["attention_mask"].sum(dim=1).tolist()
        for text, n_tokens in zip(decoded, prompt_tokens):
            results.append(
                {
                    "prediction": first_line(text),
                    "raw_output": text,
                    "latency_ms": latency_ms,
                    "prompt_tokens": int(n_tokens),
                }
            )
    return results
