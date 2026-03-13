"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_numerical_df() -> pd.DataFrame:
    """Small numerical-only DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "feat_a": np.random.randn(100),
        "feat_b": np.random.randn(100) * 2 + 1,
        "feat_c": np.random.rand(100) * 10,
        "target": np.random.randint(0, 2, 100),
    })


@pytest.fixture
def sample_metadata() -> dict:
    """Metadata for sample_numerical_df."""
    return {
        "dataset_name": "test_dataset",
        "n_samples": 100,
        "n_features": 3,
        "task_type": "classification",
        "target_col": "target",
        "numerical_features": ["feat_a", "feat_b", "feat_c"],
        "categorical_features": [],
        "feature_types": {
            "feat_a": "numerical",
            "feat_b": "numerical",
            "feat_c": "numerical",
        },
    }


@pytest.fixture
def sample_mixed_df() -> pd.DataFrame:
    """Small mixed-type DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "num_1": np.random.randn(50),
        "num_2": np.random.rand(50) * 100,
        "cat_1": np.random.choice(["A", "B", "C"], 50),
        "cat_2": np.random.choice(["X", "Y"], 50),
        "target": np.random.randint(0, 2, 50),
    })


@pytest.fixture
def mixed_metadata() -> dict:
    """Metadata for sample_mixed_df."""
    return {
        "dataset_name": "test_mixed",
        "n_samples": 50,
        "n_features": 4,
        "task_type": "classification",
        "target_col": "target",
        "numerical_features": ["num_1", "num_2"],
        "categorical_features": ["cat_1", "cat_2"],
        "feature_types": {
            "num_1": "numerical",
            "num_2": "numerical",
            "cat_1": "categorical",
            "cat_2": "categorical",
        },
    }
