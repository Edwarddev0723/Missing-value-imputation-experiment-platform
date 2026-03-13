"""Tests for imputation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.methods.statistical.mean import MeanImputer
from src.methods.statistical.knn import KNNImputer
from src.methods.statistical.softimpute import SoftImputeImputer
from src.methods.ml.missforest import MissForestImputer
from src.registry import METHOD_REGISTRY


class TestMethodRegistry:
    def test_mean_registered(self):
        assert "mean" in METHOD_REGISTRY

    def test_knn_registered(self):
        assert "knn" in METHOD_REGISTRY

    def test_missforest_registered(self):
        assert "missforest" in METHOD_REGISTRY

    def test_softimpute_registered(self):
        assert "softimpute" in METHOD_REGISTRY


def _make_missing(df: pd.DataFrame, rate: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Helper: inject random NaNs into a numeric DataFrame."""
    rng = np.random.RandomState(seed)
    mask = rng.rand(*df.shape) < rate
    df_missing = df.copy().astype(float)
    values = df_missing.values.copy()
    values[mask] = np.nan
    return pd.DataFrame(values, columns=df_missing.columns, index=df_missing.index)


class TestMeanImputer:
    def test_fit_impute(self, sample_numerical_df, sample_metadata):
        features = sample_numerical_df.drop(columns=["target"])
        df_missing = _make_missing(features)

        imp = MeanImputer()
        imp.fit(features, sample_metadata)
        result = imp.impute(df_missing)

        assert result.shape == features.shape
        assert not result.isna().any().any(), "No NaN should remain after imputation"

    def test_get_params(self):
        imp = MeanImputer()
        params = imp.get_params()
        assert params["name"] == "mean"
        assert params["family"] == "statistical"


class TestKNNImputer:
    def test_fit_impute(self, sample_numerical_df, sample_metadata):
        features = sample_numerical_df.drop(columns=["target"])
        df_missing = _make_missing(features)

        imp = KNNImputer(n_neighbors=3)
        imp.fit(df_missing, sample_metadata)
        result = imp.impute(df_missing)

        assert result.shape == features.shape
        assert not result.isna().any().any()

    def test_custom_neighbors(self):
        imp = KNNImputer(n_neighbors=10)
        assert imp.n_neighbors == 10


class TestMissForestImputer:
    def test_fit_impute(self, sample_numerical_df, sample_metadata):
        features = sample_numerical_df.drop(columns=["target"])
        df_missing = _make_missing(features)

        imp = MissForestImputer(n_estimators=10, max_iter=3)
        imp.fit(df_missing, sample_metadata)
        result = imp.impute(df_missing)

        assert result.shape == features.shape
        assert not result.isna().any().any()


class TestSoftImputeImputer:
    def test_fit_impute(self, sample_numerical_df, sample_metadata):
        features = sample_numerical_df.drop(columns=["target"])
        df_missing = _make_missing(features)

        imp = SoftImputeImputer(max_rank=3, max_iter=20)
        imp.fit(features, sample_metadata)
        result = imp.impute(df_missing)

        assert result.shape == features.shape
        assert not result.isna().any().any()


class TestUnifiedInterface:
    """Test that all imputers follow the same API."""

    @pytest.mark.parametrize("method_name", ["mean", "knn", "missforest", "softimpute"])
    def test_api_compliance(self, method_name, sample_numerical_df, sample_metadata):
        features = sample_numerical_df.drop(columns=["target"])
        df_missing = _make_missing(features)

        imputer = METHOD_REGISTRY.create(method_name)
        assert hasattr(imputer, "fit")
        assert hasattr(imputer, "impute")
        assert hasattr(imputer, "fit_impute")
        assert hasattr(imputer, "get_params")
        assert hasattr(imputer, "set_seed")

        imputer.set_seed(42)
        imputer.fit(df_missing, sample_metadata)
        result = imputer.impute(df_missing)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == features.shape
        assert not result.isna().any().any()
