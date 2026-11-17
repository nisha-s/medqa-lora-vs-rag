"""Data preparation tests: dedup, ambiguity and no symptom set shared across splits."""

import numpy as np
import pandas as pd
import pytest

from medqa import data

SYMPTOMS = [
    "cough",
    "fever",
    "rash",
    "headache",
    "nausea",
    "fatigue",
    "chills",
    "sweating",
]
SIZES = {"train_size": 40, "val_size": 10, "test_size": 20}


def make_raw(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Synthetic CSV-like frame with duplicates and ambiguous symptom sets."""
    rng = np.random.default_rng(seed)
    matrix = rng.integers(0, 2, size=(n_rows, len(SYMPTOMS)), dtype=np.int8)
    matrix[matrix.sum(axis=1) == 0, 0] = 1  # every row has at least one symptom
    diseases = rng.choice(["flu", "cold", "measles", "migraine"], size=n_rows)
    frame = pd.DataFrame(matrix, columns=SYMPTOMS)
    frame.insert(0, "diseases", diseases)
    # Exact duplicates and a forced ambiguous set.
    frame = pd.concat([frame, frame.head(30)], ignore_index=True)
    frame.loc[len(frame)] = ["flu", 1, 1, 0, 0, 0, 0, 0, 0]
    frame.loc[len(frame)] = ["cold", 1, 1, 0, 0, 0, 0, 0, 0]
    return frame


@pytest.fixture
def cases() -> pd.DataFrame:
    return data.deduplicate(data.rows_to_cases(make_raw()))


def test_rows_to_cases_keeps_column_order() -> None:
    raw = pd.DataFrame({"diseases": ["x"], "cough": [0], "fever": [1], "rash": [1]})
    cases = data.rows_to_cases(raw)
    assert cases.loc[0, "symptoms"] == ("fever", "rash")
    assert cases.loc[0, "symptom_text"] == "fever, rash"
    assert cases.loc[0, "symptom_key"] == "fever|rash"


def test_deduplicate_removes_exact_duplicates() -> None:
    cases = data.rows_to_cases(make_raw())
    deduped = data.deduplicate(cases)
    assert len(deduped) < len(cases)
    assert not deduped.duplicated(subset=["symptom_key", "disease"]).any()


def test_ambiguous_symptom_sets_are_counted_not_dropped(cases: pd.DataFrame) -> None:
    assert data.count_ambiguous(cases) >= 1
    forced = cases[cases["symptom_key"] == "cough|fever"]
    assert set(forced["disease"]) >= {"flu", "cold"}
    assert 0 < data.accuracy_ceiling(cases) < 1


def test_no_symptom_set_in_more_than_one_split(cases: pd.DataFrame) -> None:
    splits = data.split_by_symptom_set(cases, seed=42, **SIZES)
    keys = {name: set(frame["symptom_key"]) for name, frame in splits.items()}
    assert keys["train"].isdisjoint(keys["val"])
    assert keys["train"].isdisjoint(keys["test"])
    assert keys["val"].isdisjoint(keys["test"])
    assert keys["train"] <= keys["train_pool"]
    assert keys["train_pool"].isdisjoint(keys["test"] | keys["val"])


def test_split_sizes_count_symptom_sets(cases: pd.DataFrame) -> None:
    splits = data.split_by_symptom_set(cases, seed=42, **SIZES)
    assert splits["train"]["symptom_key"].nunique() == SIZES["train_size"]
    assert splits["val"]["symptom_key"].nunique() == SIZES["val_size"]
    assert splits["test"]["symptom_key"].nunique() == SIZES["test_size"]
    # Rows can exceed sets only because ambiguous sets keep all their rows.
    for frame in splits.values():
        assert len(frame) >= frame["symptom_key"].nunique()


def test_val_and_test_do_not_depend_on_train_size(cases: pd.DataFrame) -> None:
    small = data.split_by_symptom_set(cases, seed=42, **SIZES)
    big = data.split_by_symptom_set(cases, seed=42, **{**SIZES, "train_size": 60})
    for name in ("val", "test"):
        pd.testing.assert_frame_equal(small[name], big[name])


def test_split_is_deterministic_for_seed(cases: pd.DataFrame) -> None:
    first = data.split_by_symptom_set(cases, seed=7, **SIZES)
    second = data.split_by_symptom_set(cases, seed=7, **SIZES)
    other = data.split_by_symptom_set(cases, seed=8, **SIZES)
    pd.testing.assert_frame_equal(first["test"], second["test"])
    assert set(first["test"]["symptom_key"]) != set(other["test"]["symptom_key"])


def test_stratified_take_is_exact_in_total() -> None:
    sets = pd.DataFrame(
        {
            "symptom_key": [str(i) for i in range(50)],
            "strat": ["a"] * 30 + ["b"] * 15 + ["c"] * 5,
        }
    )
    taken, rest = data.stratified_take(sets, 10)
    assert len(taken) == 10 and len(rest) == 40
    assert taken["strat"].value_counts().to_dict() == {"a": 6, "b": 3, "c": 1}
    with pytest.raises(ValueError):
        data.stratified_take(sets, 51)


def test_save_and_load_round_trip(cases: pd.DataFrame, tmp_path) -> None:
    splits = data.split_by_symptom_set(cases, seed=42, **SIZES)
    data.save_splits(splits, tmp_path, extra={"seed": 42})
    loaded = data.load_splits(tmp_path)
    for name in data.SPLIT_NAMES:
        assert list(loaded[name]["symptoms"]) == list(splits[name]["symptoms"])
        assert list(loaded[name]["disease"]) == list(splits[name]["disease"])
    assert (tmp_path / "stats.json").exists()
