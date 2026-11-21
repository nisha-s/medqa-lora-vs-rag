"""LoRA fine-tuning with PEFT. Resumes from the last checkpoint if one exists.

    python -m medqa.train_lora --config configs/base.yaml --seed 42

Outputs, under output_dir:
    adapters/<model>_seed<seed>/            the PEFT adapter plus train_stats.json
    checkpoints/<model>_seed<seed>/         Trainer checkpoints used for resuming
"""

import argparse
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import pandas as pd

from medqa.config import load_config, pick_device
from medqa.data import load_splits
from medqa.prompts import training_example

log = logging.getLogger(__name__)

IGNORE_INDEX = -100  # label value the cross-entropy loss ignores


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def adapter_dir(output_dir: str | Path, model_name: str, seed: int) -> Path:
    """Where the adapter for a given model and seed is stored."""
    return Path(output_dir) / "adapters" / f"{model_slug(model_name)}_seed{seed}"


def checkpoint_dir(output_dir: str | Path, model_name: str, seed: int) -> Path:
    """Where Trainer checkpoints for a given model and seed go."""
    return Path(output_dir) / "checkpoints" / f"{model_slug(model_name)}_seed{seed}"


def tokenize_example(
    tokenizer, prompt: str, completion: str, max_len: int, loss_on_prompt: bool
) -> dict[str, list[int]]:
    """Tokenize prompt + completion + eos, masking the prompt in the labels by default.

    The prompt is tokenized separately only to find out how long it is. The ids
    themselves come from tokenizing prompt and completion together, which is
    exactly what the model sees at inference time.
    """
    prompt_len = len(tokenizer(prompt)["input_ids"])
    ids = tokenizer(prompt + completion)["input_ids"] + [tokenizer.eos_token_id]
    ids = ids[:max_len]
    labels = list(ids)
    if not loss_on_prompt:
        labels[:prompt_len] = [IGNORE_INDEX] * min(prompt_len, len(labels))
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def tokenize_split(
    tokenizer, frame: pd.DataFrame, max_len: int, loss_on_prompt: bool
) -> list[dict]:
    """Tokenize a split and warn if any example was truncated."""
    examples = []
    truncated = 0
    for text, disease in zip(frame["symptom_text"], frame["disease"]):
        prompt, completion = training_example(text, disease)
        example = tokenize_example(
            tokenizer, prompt, completion, max_len, loss_on_prompt
        )
        if example["labels"][-1] != tokenizer.eos_token_id:
            truncated += 1
        examples.append(example)
    if truncated:
        log.warning(
            "%d of %d examples lost their answer to truncation at %d tokens",
            truncated,
            len(frame),
            max_len,
        )
    return examples


class ListDataset:
    """Minimal torch-style dataset over a list of tokenized dicts."""

    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        return self.examples[i]


def build_model(model_name: str, lora_cfg: dict[str, Any], device: str):
    """Load the base model in the training dtype and wrap it with a LoRA adapter."""
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    from medqa.generate import pick_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=pick_dtype(device))
    config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def train(config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Fine-tune, save the adapter and return training stats (time, peak memory)."""
    import torch
    from transformers import (
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    from transformers.trainer_utils import get_last_checkpoint

    model_name = config["model"]["name"]
    lora_cfg = config["lora"]
    out_adapter = adapter_dir(config["output_dir"], model_name, seed)
    out_ckpt = checkpoint_dir(config["output_dir"], model_name, seed)
    stats_path = out_adapter / "train_stats.json"
    if stats_path.exists():
        log.info("adapter already trained at %s, skipping", out_adapter)
        return json.loads(stats_path.read_text())

    device = pick_device()
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    splits = load_splits(config["data"]["splits_dir"], names=("train", "val"))
    train_set = ListDataset(
        tokenize_split(
            tokenizer,
            splits["train"],
            lora_cfg["max_seq_len"],
            lora_cfg["loss_on_prompt"],
        )
    )
    val_set = ListDataset(
        tokenize_split(
            tokenizer,
            splits["val"],
            lora_cfg["max_seq_len"],
            lora_cfg["loss_on_prompt"],
        )
    )

    model = build_model(model_name, lora_cfg, device)

    wandb_cfg = config["wandb"]
    run_name = f"train_lora_{model_slug(model_name)}_seed{seed}"
    if wandb_cfg["enabled"]:
        os.environ["WANDB_PROJECT"] = wandb_cfg["project"]
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = wandb_cfg["entity"]

    args = TrainingArguments(
        output_dir=str(out_ckpt),
        seed=seed,
        data_seed=seed,
        num_train_epochs=lora_cfg["epochs"],
        max_steps=lora_cfg.get("max_steps", -1),
        per_device_train_batch_size=lora_cfg["batch_size"],
        per_device_eval_batch_size=lora_cfg["batch_size"] * 2,
        learning_rate=lora_cfg["learning_rate"],
        optim="adamw_torch",
        bf16=device == "cuda",
        use_cpu=device == "cpu",
        logging_steps=lora_cfg.get("logging_steps", 50),
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=lora_cfg.get("save_steps", 200),
        save_total_limit=2,
        report_to="wandb" if wandb_cfg["enabled"] else "none",
        run_name=run_name,
        remove_unused_columns=False,
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        processing_class=tokenizer,
    )

    last_checkpoint = get_last_checkpoint(out_ckpt) if out_ckpt.exists() else None
    if last_checkpoint:
        log.info("resuming from %s", last_checkpoint)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=last_checkpoint)
    wall_clock_s = time.perf_counter() - started

    out_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_adapter)
    eval_metrics = trainer.evaluate()
    stats = {
        "model": model_name,
        "seed": seed,
        "device": device,
        "train_rows": len(train_set),
        "val_rows": len(val_set),
        "steps": result.global_step,
        "epochs": lora_cfg["epochs"],
        "train_loss": result.training_loss,
        "val_loss": eval_metrics.get("eval_loss"),
        "wall_clock_s": wall_clock_s,
        "resumed_from": last_checkpoint,
        "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1e9
        if device == "cuda"
        else None,
        "lora": lora_cfg,
        "log_history": trainer.state.log_history,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(
        "saved adapter to %s (%.0f s, %d steps, train loss %.4f)",
        out_adapter,
        wall_clock_s,
        result.global_step,
        result.training_loss,
    )

    # Adapter is on disk now, so the resume checkpoints have done their job.
    shutil.rmtree(out_ckpt, ignore_errors=True)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, help="defaults to config.seed")
    cli = parser.parse_args()
    cfg = load_config(cli.config)
    train(cfg, cli.seed if cli.seed is not None else cfg["seed"])
