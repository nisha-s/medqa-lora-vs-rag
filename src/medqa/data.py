"""Load the symptoms CSV, deduplicate, split by symptom set and save to disk.

Run as a module to build the splits:
    python -m medqa.data --config configs/base.yaml
"""

from pathlib import Path
from typing import Any

import pandas as pd


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    """Read the raw CSV with binary symptom columns and a disease column."""
    raise NotImplementedError


def rows_to_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Turn each row into (symptom set as sorted tuple, disease)."""
    raise NotImplementedError


def deduplicate(cases: pd.DataFrame) -> pd.DataFrame:
    """Drop exact (symptom set, disease) duplicates."""
    raise NotImplementedError


def count_ambiguous(cases: pd.DataFrame) -> int:
    """Count symptom sets that map to more than one disease (accuracy ceiling)."""
    raise NotImplementedError


def split_by_symptom_set(
    cases: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Split so no symptom set appears in more than one split; stratify by disease."""
    raise NotImplementedError


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write each split to CSV plus a small JSON of counts and the ambiguity count."""
    raise NotImplementedError


def load_splits(splits_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Read previously saved splits."""
    raise NotImplementedError


def main(config: dict[str, Any]) -> None:
    """End-to-end data preparation from config."""
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented yet")
