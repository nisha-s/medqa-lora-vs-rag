"""Small YAML config loader.

A config can name a parent with `extends:` and inherit from it (one level deep,
though parents can chain). Anywhere a value contains ${output_dir} it gets
swapped for the resolved output directory, so paths stay in one place.
"""

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merging in its `extends` parent if present."""
    cfg = _load_with_parents(Path(path))
    # Expand after the merge, so a child that overrides output_dir still wins.
    return _expand_output_dir(cfg, cfg["output_dir"])


def _load_with_parents(path: Path) -> dict[str, Any]:
    """Read one YAML file and merge it over its `extends` parent, recursively."""
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    parent_name = cfg.pop("extends", None)
    if parent_name:
        parent = _load_with_parents(path.parent / parent_name)
        cfg = merge(parent, cfg)
    return cfg


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_output_dir(value: Any, output_dir: str) -> Any:
    """Replace the literal ${output_dir} in every string value."""
    if isinstance(value, dict):
        return {k: _expand_output_dir(v, output_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_output_dir(v, output_dir) for v in value]
    if isinstance(value, str):
        return value.replace("${output_dir}", output_dir)
    return value


def pick_device() -> str:
    """Return "cuda", "mps" or "cpu", whichever is available first."""
    import torch  # kept local so data prep can run without torch installed

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
