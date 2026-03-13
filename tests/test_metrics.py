"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.imputation_metrics import ImputationEvaluator
from src.evaluation.downstream_metrics import DownstreamEvaluator


class TestImputationEvaluator:
    def test_numerical_metrics(self):
        X_true = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        X_imputed = pd.DataFrame({
            "a": [1.0, 2.5, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 35.0, 40.0, 50.0],
        })
        mask = np.array([
            [False, False],
            [True, False],
            [False, True],
            [False, False],
            [False, False],
        ])
        metadata = {
            "numerical_features": ["a", "b"],
            "categorical_features": [],
        }

        evaluator = ImputationEvaluator()
        result = evaluator.evaluate(X_true, X_imputed, mask, metadata)

        assert "numerical_rmse" in result
        assert "numerical_mae" in result
        assert result["numerical_rmse"] > 0
        assert result["numerical_mae"] > 0

    def test_perfect_imputation(self):
        X_true = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        X_imputed = X_true.copy()
        mask = np.array([[True, False], [False, True], [False, False]])
        metadata = {"numerical_features": ["a", "b"], "categorical_features": []}

        evaluator = ImputationEvaluator()
        result = evaluator.evaluate(X_true, X_imputed, mask, metadata)

        assert result["numerical_rmse"] == 0.0
        assert result["numerical_mae"] == 0.0

    def test_no_missing_returns_empty(self):
        X_true = pd.DataFrame({"a": [1.0, 2.0]})
        mask = np.array([[False], [False]])
        metadata = {"numerical_features": ["a"], "categorical_features": []}

        evaluator = ImputationEvaluator()
        result = evaluator.evaluate(X_true, X_true, mask, metadata)

        # No missing values → no numerical metrics
        assert "numerical_rmse" not in result


class TestDownstreamEvaluator:
    def test_classification(self):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y_train = pd.Series(np.random.randint(0, 2, 100))
        X_test = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y_test = pd.Series(np.random.randint(0, 2, 30))

        evaluator = DownstreamEvaluator()
        result = evaluator.fit_and_evaluate(
            X_train, y_train, X_test, y_test, task_type="classification"
        )

        assert "downstream_accuracy" in result
        assert "downstream_f1" in result
        assert 0 <= result["downstream_accuracy"] <= 1

    def test_regression(self):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(100, 3), columns=["a", "b", "c"])
        y_train = pd.Series(np.random.randn(100))
        X_test = pd.DataFrame(np.random.randn(30, 3), columns=["a", "b", "c"])
        y_test = pd.Series(np.random.randn(30))

        evaluator = DownstreamEvaluator()
        result = evaluator.fit_and_evaluate(
            X_train, y_train, X_test, y_test, task_type="regression"
        )

        assert "downstream_rmse" in result
        assert "downstream_r2" in result
