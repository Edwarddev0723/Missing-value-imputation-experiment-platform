"""Imputation quality metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.constants import CATEGORICAL, NUMERICAL
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ImputationEvaluator:
    """Evaluate imputation quality on masked positions only."""

    def evaluate(
        self,
        X_true: pd.DataFrame,
        X_imputed: pd.DataFrame,
        missing_mask: np.ndarray,
        metadata: dict[str, Any],
    ) -> dict[str, float]:
        """Compute imputation metrics only on positions that were missing.

        Args:
            X_true: Ground-truth complete data.
            X_imputed: Imputed data.
            missing_mask: Boolean array, True where values were missing.
            metadata: Dataset metadata with feature type info.

        Returns:
            Dict of metric_name -> value.
        """
        results: dict[str, float] = {}

        numerical_cols = metadata.get("numerical_features", [])
        categorical_cols = metadata.get("categorical_features", [])

        # Numerical metrics
        num_cols_present = [c for c in numerical_cols if c in X_true.columns]
        if num_cols_present:
            true_num = X_true[num_cols_present].values
            imputed_num = X_imputed[num_cols_present].values
            mask_num = missing_mask[:, [X_true.columns.get_loc(c) for c in num_cols_present]]

            if mask_num.any():
                true_vals = true_num[mask_num]
                pred_vals = imputed_num[mask_num]
                rmse = float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))
                mae = float(np.mean(np.abs(true_vals - pred_vals)))
                # NRMSE: normalized by std of true values
                std = float(np.std(true_vals))
                nrmse = rmse / std if std > 0 else rmse
                results["numerical_rmse"] = rmse
                results["numerical_mae"] = mae
                results["numerical_nrmse"] = nrmse

        # Categorical metrics
        cat_cols_present = [c for c in categorical_cols if c in X_true.columns]
        if cat_cols_present:
            true_cat = X_true[cat_cols_present].values
            imputed_cat = X_imputed[cat_cols_present].values
            mask_cat = missing_mask[:, [X_true.columns.get_loc(c) for c in cat_cols_present]]

            if mask_cat.any():
                true_vals = true_cat[mask_cat]
                pred_vals = imputed_cat[mask_cat]
                acc = float(accuracy_score(true_vals, pred_vals))
                results["categorical_accuracy"] = acc

        # Overall: combined metric across all missing positions
        all_mask = missing_mask.any()
        if missing_mask.any():
            n_missing = int(missing_mask.sum())
            results["n_missing_values"] = n_missing

        return results
