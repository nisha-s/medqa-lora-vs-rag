"""Load the symptoms CSV, deduplicate, split by symptom set and save to disk.

Split sizes count unique symptom sets. A symptom set that maps to several
diseases contributes all of its rows to the same split, so row counts can be
slightly larger than the configured sizes; both are written to stats.json.

Run as a module to build the splits:
    python -m medqa.data --config configs/base.yaml
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from medqa.config import load_config
from medqa.prompts import symptoms_to_text

log = logging.getLogger(__name__)

LABEL_COL = "diseases"
KEY_SEP = "|"
# Treat ambiguous sets as one stratum of their own, so every split ends up with
# a similar share of them and the ceilings stay comparable across splits.
AMBIGUOUS = "__ambiguous__"
SPLIT_NAMES = ("train", "val", "test", "train_pool")


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV with binary symptom columns and a disease column."""
    header = pd.read_csv(csv_path, nrows=0).columns
    label_col = LABEL_COL if LABEL_COL in header else header[0]
    dtypes = {col: "int8" for col in header if col != label_col}
    return pd.read_csv(csv_path, dtype=dtypes)


def rows_to_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Turn each row into (symptom tuple, key, symptom text, disease).

    Symptom order follows the CSV columns, same as the original notebook.
    """
    label_col = LABEL_COL if LABEL_COL in df.columns else df.columns[0]
    symptom_cols = [c for c in df.columns if c != label_col]
    names = np.array([c.strip() for c in symptom_cols], dtype=object)
    active = df[symptom_cols].to_numpy(dtype=np.int8) != 0

    symptoms = [tuple(names[row]) for row in active]
    return pd.DataFrame(
        {
            "symptoms": symptoms,
            "symptom_key": [KEY_SEP.join(s) for s in symptoms],
            "symptom_text": [symptoms_to_text(s) for s in symptoms],
            "disease": df[label_col].astype(str).str.strip().to_numpy(),
        }
    )


def deduplicate(cases: pd.DataFrame) -> pd.DataFrame:
    """Drop exact (symptom set, disease) duplicates."""
    return cases.drop_duplicates(subset=["symptom_key", "disease"]).reset_index(
        drop=True
    )


def count_ambiguous(cases: pd.DataFrame) -> int:
    """Count symptom sets that map to more than one disease."""
    diseases_per_set = cases.groupby("symptom_key")["disease"].nunique()
    return int((diseases_per_set > 1).sum())


def accuracy_ceiling(cases: pd.DataFrame) -> float:
    """Best possible exact-match accuracy: one answer per symptom set."""
    if len(cases) == 0:
        return 0.0
    return cases["symptom_key"].nunique() / len(cases)


def unique_symptom_sets(cases: pd.DataFrame) -> pd.DataFrame:
    """One row per symptom set with its stratification label."""
    diseases = cases.groupby("symptom_key")["disease"].agg(["nunique", "first"])
    strat = np.where(diseases["nunique"] > 1, AMBIGUOUS, diseases["first"])
    return pd.DataFrame({"symptom_key": diseases.index, "strat": strat}).reset_index(
        drop=True
    )


def stratified_take(sets: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take n rows, spread across strata in proportion to each stratum's size.

    `sets` has to be shuffled already. Rounding is largest-remainder, so the
    total is exactly n even though a small stratum can come away with nothing.
    """
    if n > len(sets):
        raise ValueError(f"asked for {n} symptom sets but only {len(sets)} remain")
    counts = sets["strat"].value_counts()
    quota = counts * n / len(sets)
    take = np.floor(quota).astype(int)
    leftover = n - int(take.sum())
    fractional = (quota - take).sort_values(ascending=False, kind="mergesort")
    take[fractional.index[:leftover]] += 1

    taken_index: list[int] = []
    for strat, k in take.items():
        if k > 0:
            taken_index.extend(sets.index[sets["strat"] == strat][:k])
    taken = sets.loc[taken_index]
    rest = sets.drop(taken_index)
    return taken, rest


def split_by_symptom_set(
    cases: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split so no symptom set appears in more than one split.

    Test and val are taken first, so they are identical for any train_size.
    """
    sets = unique_symptom_sets(cases).sample(frac=1, random_state=seed)
    sets = sets.reset_index(drop=True)
    test, rest = stratified_take(sets, test_size)
    val, train_pool = stratified_take(rest, val_size)
    train, _ = stratified_take(train_pool, train_size)

    chosen = {"train": train, "val": val, "test": test, "train_pool": train_pool}
    splits = {}
    for name, chosen_sets in chosen.items():
        mask = cases["symptom_key"].isin(chosen_sets["symptom_key"])
        splits[name] = cases[mask].reset_index(drop=True)
    return splits


def split_stats(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Row counts, symptom-set counts, ambiguity and ceiling per split."""
    stats = {}
    for name, frame in splits.items():
        stats[name] = {
            "rows": len(frame),
            "symptom_sets": int(frame["symptom_key"].nunique()),
            "diseases": int(frame["disease"].nunique()),
            "ambiguous_symptom_sets": count_ambiguous(frame),
            "accuracy_ceiling": round(accuracy_ceiling(frame), 4),
        }
    return stats


def save_splits(
    splits: dict[str, pd.DataFrame], out_dir: str | Path, extra: dict
) -> None:
    """Write each split to CSV plus stats.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.drop(columns=["symptoms"]).to_csv(out_dir / f"{name}.csv", index=False)
    stats = {"splits": split_stats(splits), **extra}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))


def load_splits(
    splits_dir: str | Path, names: tuple[str, ...] = SPLIT_NAMES
) -> dict[str, pd.DataFrame]:
    """Read previously saved splits and rebuild the symptom tuples."""
    splits = {}
    for name in names:
        frame = pd.read_csv(
            Path(splits_dir) / f"{name}.csv", dtype=str, keep_default_na=False
        )
        frame["symptoms"] = [
            tuple(k.split(KEY_SEP)) if k else () for k in frame["symptom_key"]
        ]
        splits[name] = frame
    return splits


def main(config: dict[str, Any]) -> None:
    """End-to-end data preparation from config."""
    data_cfg = config["data"]
    log.info("reading %s", data_cfg["csv_path"])
    raw = load_raw(data_cfg["csv_path"])
    cases = rows_to_cases(raw)
    n_raw = len(cases)

    cases = deduplicate(cases)
    n_ambiguous = count_ambiguous(cases)
    log.info("rows: %d raw, %d after dedup", n_raw, len(cases))
    log.info("symptom sets mapping to more than one disease: %d", n_ambiguous)

    splits = split_by_symptom_set(
        cases,
        train_size=data_cfg["train_size"],
        val_size=data_cfg["val_size"],
        test_size=data_cfg["test_size"],
        seed=config["seed"],
    )
    extra = {
        "seed": config["seed"],
        "raw_rows": n_raw,
        "dedup_rows": len(cases),
        "ambiguous_symptom_sets_total": n_ambiguous,
    }
    save_splits(splits, data_cfg["splits_dir"], extra)
    for name, s in split_stats(splits).items():
        log.info("%-10s %s", name, s)
    log.info("saved to %s", data_cfg["splits_dir"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    main(load_config(parser.parse_args().config))
