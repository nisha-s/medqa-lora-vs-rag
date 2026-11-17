"""Run an experiment arm, save per-example predictions and aggregate results.

python -m medqa.evaluate --config configs/base.yaml --arm retrieval_only --k 5
python -m medqa.evaluate --config configs/base.yaml --arm base_rag --k 5
python -m medqa.evaluate --config configs/base.yaml --arm lora --seed 42
python -m medqa.evaluate --config configs/base.yaml --mmlu
python -m medqa.evaluate --config configs/base.yaml --aggregate
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medqa import metrics
from medqa.config import load_config, pick_device
from medqa.data import accuracy_ceiling, load_splits
from medqa.generate import generate_batch, load_model
from medqa.prompts import rag_prompt, zero_shot_prompt
from medqa.retrieval import (
    build_retriever,
    majority_label,
    recall_at_k,
    retrieved_cases,
)
from medqa.train_lora import adapter_dir

log = logging.getLogger(__name__)

ARMS = ("base_zero_shot", "base_rag", "lora", "lora_rag", "retrieval_only")
RAG_ARMS = ("base_rag", "lora_rag", "retrieval_only")
LORA_ARMS = ("lora", "lora_rag")
METRIC_COLS = (
    "exact_match",
    "recall_at_k",
    "semantic",
    "rouge_l",
    "latency_ms",
    "prompt_tokens",
)


def run_name(arm: str, retriever: str | None, k: int | None, seed: int | None) -> str:
    """Stable file stem for one run, e.g. base_rag_jaccard_k5 or lora_seed42."""
    parts = [arm]
    if retriever:
        parts.append(retriever)
    if k is not None:
        parts.append(f"k{k}")
    if seed is not None:
        parts.append(f"seed{seed}")
    return "_".join(parts)


def retrieve(
    config: dict[str, Any],
    train: pd.DataFrame,
    queries: pd.DataFrame,
    retriever_name: str,
    k: int,
):
    """Neighbour indices (n_queries, k) and the per-query retrieval time in ms."""
    retriever = build_retriever(config, name=retriever_name)
    retriever.build(train)
    start = time.perf_counter()
    neighbours = retriever.query_batch(queries, k)
    latency_ms = (time.perf_counter() - start) * 1000 / len(queries)
    return neighbours, latency_ms


def run_retrieval_only(
    config: dict[str, Any],
    train: pd.DataFrame,
    queries: pd.DataFrame,
    retriever_name: str,
    k: int,
) -> pd.DataFrame:
    """Predict the majority disease among the k nearest training cases. No LLM."""
    neighbours, latency_ms = retrieve(config, train, queries, retriever_name, k)
    rows = []
    for i, disease in enumerate(queries["disease"]):
        labels = train["disease"].iloc[neighbours[i]].tolist()
        rows.append(
            {
                "prediction": majority_label(labels),
                "recall_at_k": recall_at_k(labels, disease),
                "retrieved_labels": " | ".join(labels),
                "latency_ms": latency_ms,
                "prompt_tokens": 0,
            }
        )
    return pd.DataFrame(rows)


def run_llm(
    config: dict[str, Any],
    train: pd.DataFrame,
    queries: pd.DataFrame,
    adapter_path: str | None,
    retriever_name: str | None,
    k: int | None,
) -> pd.DataFrame:
    """Generate a disease for each query, zero-shot or with retrieved examples."""
    model_cfg = config["model"]
    if retriever_name is not None:
        neighbours, retrieval_ms = retrieve(config, train, queries, retriever_name, k)
        prompts, recalls, retrieved_labels = [], [], []
        for i, (text, disease) in enumerate(
            zip(queries["symptom_text"], queries["disease"])
        ):
            cases = retrieved_cases(train, neighbours[i])
            prompts.append(rag_prompt(cases, text))
            labels = [label for _, label in cases]
            recalls.append(recall_at_k(labels, disease))
            retrieved_labels.append(" | ".join(labels))
    else:
        retrieval_ms = 0.0
        prompts = [zero_shot_prompt(text) for text in queries["symptom_text"]]
        recalls = [np.nan] * len(queries)
        retrieved_labels = [""] * len(queries)

    model, tokenizer = load_model(model_cfg["name"], adapter_path, pick_device())
    outputs = generate_batch(
        model,
        tokenizer,
        prompts,
        max_new_tokens=model_cfg["max_new_tokens"],
        batch_size=model_cfg["eval_batch_size"],
        max_prompt_tokens=model_cfg["max_prompt_tokens"],
    )
    frame = pd.DataFrame(outputs)
    frame["latency_ms"] += retrieval_ms
    frame["recall_at_k"] = recalls
    frame["retrieved_labels"] = retrieved_labels
    frame["prompt"] = prompts
    return frame


def score(
    config: dict[str, Any], queries: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Attach the query, its label and every per-example metric."""
    frame = predictions.copy()
    frame.insert(0, "disease", queries["disease"].to_numpy())
    frame.insert(0, "symptom_text", queries["symptom_text"].to_numpy())
    frame["exact_match"] = [
        metrics.exact_match(p, t) for p, t in zip(frame["prediction"], frame["disease"])
    ]
    frame["rouge_l"] = [
        metrics.rouge_l(p, t) for p, t in zip(frame["prediction"], frame["disease"])
    ]

    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(
        config["retrieval"]["embedding_model"], device=pick_device()
    )
    frame["semantic"] = metrics.semantic_similarity(
        frame["prediction"].tolist(), frame["disease"].tolist(), embedder
    )
    return frame


def summarize(scored: pd.DataFrame, queries: pd.DataFrame) -> dict[str, float]:
    """Mean of every metric column plus the split's accuracy ceiling."""
    summary = {}
    for col in METRIC_COLS:
        if col in scored and scored[col].notna().any():
            summary[col] = float(scored[col].astype(float).mean())
    summary["n"] = len(scored)
    summary["accuracy_ceiling"] = accuracy_ceiling(queries)
    return summary


def save_run(
    config: dict[str, Any], name: str, scored: pd.DataFrame, summary: dict
) -> None:
    """Write per-example predictions to CSV and the summary to JSON."""
    out = Path(config["output_dir"])
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "predictions" / f"{name}.csv", index=False)
    (out / "results" / f"{name}.json").write_text(json.dumps(summary, indent=2))
    shown = {k: round(v, 4) if isinstance(v, float) else v for k, v in summary.items()}
    log.info("%s: %s", name, shown)


def run_arm(
    config: dict[str, Any],
    arm: str,
    split: str = "test",
    seed: int | None = None,
    k: int | None = None,
    retriever: str | None = None,
) -> dict:
    """Run one arm on one split, save its outputs and return the summary."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    uses_rag = arm in RAG_ARMS
    retriever = (retriever or config["retrieval"]["retriever"]) if uses_rag else None
    k = (k or config["retrieval"]["k"]) if uses_rag else None
    if arm in LORA_ARMS:
        if seed is None:
            raise ValueError(f"arm {arm!r} needs --seed to pick an adapter")
        adapter = adapter_dir(config["output_dir"], config["model"]["name"], seed)
        if not adapter.exists():
            raise FileNotFoundError(f"no adapter at {adapter}; run train_lora first")
        adapter_path: str | None = str(adapter)
    else:
        adapter_path, seed = None, None

    splits = load_splits(config["data"]["splits_dir"], names=("train", split))
    train, queries = splits["train"], splits[split]

    if arm == "retrieval_only":
        predictions = run_retrieval_only(config, train, queries, retriever, k)
    else:
        predictions = run_llm(config, train, queries, adapter_path, retriever, k)

    scored = score(config, queries, predictions)
    summary = {
        "arm": arm,
        "split": split,
        "model": config["model"]["name"],
        "retriever": retriever,
        "k": k,
        "seed": seed,
    }
    summary.update(summarize(scored, queries))
    name = run_name(arm, retriever, k, seed)
    if split != "test":
        name = f"{name}_{split}"
    save_run(config, name, scored, summary)
    log_to_wandb(config, name, summary, scored)
    return summary


def load_mmlu_subset(config: dict[str, Any]) -> pd.DataFrame:
    """Fixed MMLU subset: N questions per subject, cached so every run scores the same ones."""
    mmlu_cfg = config["mmlu"]
    cache = Path(config["output_dir"]) / "mmlu_subset.csv"
    if cache.exists():
        frame = pd.read_csv(cache)
        frame["choices"] = frame["choices"].map(json.loads)
        return frame

    from datasets import load_dataset

    parts = []
    for subject in mmlu_cfg["subjects"]:
        test = load_dataset("cais/mmlu", subject, split="test").to_pandas()
        test["subject"] = subject
        # Seed 0 is fixed on purpose: the subset must not change with the experiment seed.
        parts.append(
            test.sample(
                n=min(mmlu_cfg["questions_per_subject"], len(test)), random_state=0
            )
        )
    frame = pd.concat(parts, ignore_index=True)[
        ["subject", "question", "choices", "answer"]
    ]
    frame["choices"] = frame["choices"].map(list)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(choices=frame["choices"].map(json.dumps)).to_csv(cache, index=False)
    return frame


def run_mmlu(config: dict[str, Any]) -> list[dict]:
    """MMLU retention for the base model and each trained LoRA adapter.

    RAG leaves the weights unchanged, so the base result applies to the RAG arms.
    """
    subset = load_mmlu_subset(config)
    questions = subset.to_dict("records")
    model_name = config["model"]["name"]
    variants: list[tuple[str, int | None, str | None]] = [("base", None, None)]
    for seed in config["seeds"]:
        adapter = adapter_dir(config["output_dir"], model_name, seed)
        if adapter.exists():
            variants.append(("lora", seed, str(adapter)))

    summaries = []
    for weights, seed, adapter_path in variants:
        model, tokenizer = load_model(model_name, adapter_path, pick_device())
        accuracy, rows = metrics.mmlu_accuracy(
            model, tokenizer, questions, config["model"]["eval_batch_size"]
        )
        scored = pd.DataFrame(rows)
        per_subject = scored.groupby("subject")["correct"].mean().round(4).to_dict()
        summary = {
            "arm": "mmlu",
            "weights": weights,
            "model": model_name,
            "seed": seed,
            "mmlu_accuracy": accuracy,
            "n": len(rows),
            "per_subject": per_subject,
        }
        name = "mmlu_base" if seed is None else f"mmlu_lora_seed{seed}"
        save_run(config, name, scored, summary)
        log_to_wandb(config, name, summary, scored)
        summaries.append(summary)
        del model
    return summaries


def aggregate(config: dict[str, Any]) -> pd.DataFrame:
    """Combine per-run summaries into one table with mean and std across seeds."""
    results_dir = Path(config["output_dir"]) / "results"
    rows = [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
    if not rows:
        raise FileNotFoundError(f"no result files in {results_dir}")
    frame = pd.DataFrame(rows)
    out = Path(config["output_dir"])

    mmlu = frame[frame["arm"] == "mmlu"]
    if len(mmlu):
        mmlu_table = _mean_std(mmlu, ["model", "weights"], ["mmlu_accuracy"])
        mmlu_table.to_csv(out / "mmlu.csv", index=False)
        log.info("MMLU retention:\n%s", mmlu_table.to_string())

    arms = frame[frame["arm"] != "mmlu"]
    metric_cols = [c for c in METRIC_COLS if c in arms]
    table = _mean_std(arms, ["arm", "split", "model", "retriever", "k"], metric_cols)
    table.to_csv(out / "results.csv", index=False)
    return table


def _mean_std(
    frame: pd.DataFrame, group_cols: list[str], metric_cols: list[str]
) -> pd.DataFrame:
    """Mean and std of each metric per group, plus the number of runs."""
    grouped = frame.groupby(group_cols, dropna=False)
    table = grouped[metric_cols].agg(["mean", "std"])
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table["n_runs"] = grouped.size()
    return table.reset_index()


def log_to_wandb(
    config: dict[str, Any],
    name: str,
    summary: dict,
    scored: pd.DataFrame,
    n_samples: int = 50,
) -> None:
    """Log the summary and a table of sample predictions; no-op when disabled."""
    if not config["wandb"]["enabled"]:
        return
    import wandb  # the API key comes from WANDB_API_KEY in the environment

    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        name=name,
        config=config,
        reinit=True,
    )
    run.log({k: v for k, v in summary.items() if isinstance(v, (int, float))})
    sample_cols = [
        c
        for c in (
            "symptom_text",
            "disease",
            "prediction",
            "exact_match",
            "retrieved_labels",
        )
        if c in scored
    ]
    run.log({"samples": wandb.Table(dataframe=scored[sample_cols].head(n_samples))})
    run.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--seed", type=int, help="adapter seed, for the LoRA arms")
    parser.add_argument("--k", type=int, help="override retrieval.k")
    parser.add_argument(
        "--retriever",
        choices=("embedding", "jaccard"),
        help="override retrieval.retriever",
    )
    parser.add_argument("--mmlu", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.arm:
        run_arm(cfg, args.arm, args.split, args.seed, args.k, args.retriever)
    if args.mmlu:
        run_mmlu(cfg)
    if args.aggregate:
        print(aggregate(cfg).to_string())
