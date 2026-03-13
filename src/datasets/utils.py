"""Dataset utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    target_col: str | None,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into train/valid/test with optional stratification.

    Returns:
        Dict with keys 'train', 'valid', 'test'.
    """
    test_ratio = 1.0 - train_ratio - valid_ratio
    assert test_ratio > 0, "train_ratio + valid_ratio must be < 1.0"

    stratify = df[target_col] if target_col and target_col in df.columns else None

    train_df, temp_df = train_test_split(
        df, test_size=(valid_ratio + test_ratio), random_state=seed, stratify=stratify
    )

    # Adjust ratio for second split
    if valid_ratio > 0:
        relative_valid = valid_ratio / (valid_ratio + test_ratio)
        stratify_temp = temp_df[target_col] if stratify is not None else None
        valid_df, test_df = train_test_split(
            temp_df, test_size=(1 - relative_valid), random_state=seed, stratify=stratify_temp
        )
    else:
        valid_df = pd.DataFrame(columns=df.columns)
        test_df = temp_df

    return {
        "train": train_df.reset_index(drop=True),
        "valid": valid_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
