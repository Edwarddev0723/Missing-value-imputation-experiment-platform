"""Tests for missingness generators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.missingness.mcar import MCARGenerator
from src.missingness.mar import MARGenerator
from src.registry import MISSINGNESS_REGISTRY


class TestMCARGenerator:
    def test_registered(self):
        assert "MCAR" in MISSINGNESS_REGISTRY

    def test_generate_basic(self, sample_numerical_df, sample_metadata):
        gen = MCARGenerator()
        features = sample_numerical_df.drop(columns=["target"])
        result = gen.generate(features, sample_metadata, seed=42, rate=0.2)

        assert result.X_missing.shape == features.shape
        assert result.missing_mask.shape == features.shape
        assert result.observed_mask.shape == features.shape
        assert result.missing_mask.any(), "Should have some missing values"
        assert (result.missing_mask == ~result.observed_mask).all()

    def test_reproducibility(self, sample_numerical_df, sample_metadata):
        gen = MCARGenerator()
        features = sample_numerical_df.drop(columns=["target"])

        r1 = gen.generate(features, sample_metadata, seed=42, rate=0.2)
        r2 = gen.generate(features, sample_metadata, seed=42, rate=0.2)

        assert (r1.missing_mask == r2.missing_mask).all(), "Same seed should give same mask"

    def test_different_seeds(self, sample_numerical_df, sample_metadata):
        gen = MCARGenerator()
        features = sample_numerical_df.drop(columns=["target"])

        r1 = gen.generate(features, sample_metadata, seed=42, rate=0.2)
        r2 = gen.generate(features, sample_metadata, seed=99, rate=0.2)

        assert not (r1.missing_mask == r2.missing_mask).all(), "Different seeds should differ"

    def test_approximate_rate(self, sample_numerical_df, sample_metadata):
        gen = MCARGenerator()
        features = sample_numerical_df.drop(columns=["target"])
        rate = 0.3
        result = gen.generate(features, sample_metadata, seed=42, rate=rate)

        actual_rate = result.missing_mask.mean()
        assert abs(actual_rate - rate) < 0.1, f"Actual rate {actual_rate} too far from {rate}"

    def test_missing_values_are_nan(self, sample_numerical_df, sample_metadata):
        gen = MCARGenerator()
        features = sample_numerical_df.drop(columns=["target"])
        result = gen.generate(features, sample_metadata, seed=42, rate=0.2)

        assert result.X_missing.isna().sum().sum() > 0


class TestMARGenerator:
    def test_registered(self):
        assert "MAR" in MISSINGNESS_REGISTRY

    def test_generate_basic(self, sample_numerical_df, sample_metadata):
        gen = MARGenerator()
        features = sample_numerical_df.drop(columns=["target"])
        result = gen.generate(features, sample_metadata, seed=42, rate=0.2)

        assert result.X_missing.shape == features.shape
        assert result.missing_mask.any()

    def test_reproducibility(self, sample_numerical_df, sample_metadata):
        gen = MARGenerator()
        features = sample_numerical_df.drop(columns=["target"])

        r1 = gen.generate(features, sample_metadata, seed=42, rate=0.2)
        r2 = gen.generate(features, sample_metadata, seed=42, rate=0.2)

        assert (r1.missing_mask == r2.missing_mask).all()
