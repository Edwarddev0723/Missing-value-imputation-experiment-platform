"""I/O utilities for saving and loading results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it does not exist and return the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Save data as JSON."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_results_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of flat result dicts as CSV."""
    path = Path(path)
    ensure_dir(path.parent)
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """Load results CSV."""
    return pd.read_csv(path)
