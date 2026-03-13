"""Tests for dataset adapters."""

from __future__ import annotations

import pytest

from src.datasets.base import BaseDataset
from src.registry import DATASET_REGISTRY

# Trigger registration
import src.datasets.adult  # noqa: F401
import src.datasets.wine  # noqa: F401
import src.datasets.heart  # noqa: F401


class TestDatasetRegistry:
    def test_adult_registered(self):
        assert "adult" in DATASET_REGISTRY

    def test_wine_registered(self):
        assert "wine" in DATASET_REGISTRY

    def test_heart_registered(self):
        assert "heart" in DATASET_REGISTRY


class TestAdultDataset:
    """Test Adult dataset adapter (downloads data on first run)."""

    @pytest.fixture
    def adult(self):
        ds = DATASET_REGISTRY.create("adult", data_dir="data/raw")
        return ds

    def test_load(self, adult):
        df = adult.load()
        assert len(df) > 0
        assert "income" in df.columns

    def test_metadata(self, adult):
        adult.load()
        meta = adult.get_metadata()
        assert meta["dataset_name"] == "adult"
        assert meta["task_type"] == "classification"
        assert len(meta["numerical_features"]) > 0

    def test_splits(self, adult):
        adult.load()
        splits = adult.get_splits(seed=42)
        assert "train" in splits
        assert "valid" in splits
        assert "test" in splits
        assert len(splits["train"]) > len(splits["test"])

    def test_get_features_excludes_target(self, adult):
        df = adult.load()
        features = adult.get_features(df)
        assert "income" not in features.columns

    def test_get_target(self, adult):
        df = adult.load()
        target = adult.get_target(df)
        assert target is not None
        assert target.name == "income"
