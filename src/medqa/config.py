"""Load YAML configs with a single level of `extends` and ${output_dir} expansion."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, merging in its `extends` parent if present."""
    raise NotImplementedError


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    raise NotImplementedError


def pick_device() -> str:
    """Return "cuda", "mps" or "cpu", whichever is available first."""
    raise NotImplementedError
