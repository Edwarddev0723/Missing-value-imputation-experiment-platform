"""Imputation quality metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

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

        # --- Imputation Quality: Numerical ---
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
                results["numerical_rmse"] = rmse
                results["numerical_mae"] = mae

        # --- Imputation Quality: Categorical ---
        cat_cols_present = [c for c in categorical_cols if c in X_true.columns]
        if cat_cols_present:
            true_cat = X_true[cat_cols_present].values
            imputed_cat = X_imputed[cat_cols_present].values
            mask_cat = missing_mask[:, [X_true.columns.get_loc(c) for c in cat_cols_present]]

            if mask_cat.any():
                true_vals = true_cat[mask_cat]
                pred_vals = imputed_cat[mask_cat]
                acc = float(accuracy_score(true_vals, pred_vals))
                macro_f1 = float(f1_score(true_vals, pred_vals, average="macro", zero_division=0))
                results["categorical_accuracy"] = acc
                results["categorical_macro_f1"] = macro_f1

        # --- Distribution Preservation: Wasserstein Distance (numerical) ---
        if num_cols_present:
            wd_list = []
            for col in num_cols_present:
                true_col = X_true[col].values.astype(float)
                imp_col = X_imputed[col].values.astype(float)
                # Use all values (not just masked) to compare full distributions
                finite = np.isfinite(true_col) & np.isfinite(imp_col)
                if finite.sum() > 1:
                    wd_list.append(wasserstein_distance(true_col[finite], imp_col[finite]))
            if wd_list:
                results["wasserstein_distance_mean"] = float(np.mean(wd_list))

        # --- Distribution Preservation: JSD (categorical) ---
        if cat_cols_present:
            jsd_list = []
            for col in cat_cols_present:
                true_col = X_true[col].values
                imp_col = X_imputed[col].values
                # Compute category distributions
                all_cats = np.union1d(np.unique(true_col), np.unique(imp_col))
                n_classes = len(all_cats)
                cat_to_idx = {c: i for i, c in enumerate(all_cats)}
                true_counts = np.zeros(n_classes)
                imp_counts = np.zeros(n_classes)
                for v in true_col:
                    true_counts[cat_to_idx[v]] += 1
                for v in imp_col:
                    imp_counts[cat_to_idx[v]] += 1
                # Normalize to probability distributions
                true_dist = true_counts / true_counts.sum() if true_counts.sum() > 0 else true_counts
                imp_dist = imp_counts / imp_counts.sum() if imp_counts.sum() > 0 else imp_counts
                jsd_val = jensenshannon(true_dist, imp_dist)
                if np.isfinite(jsd_val):
                    jsd_list.append(float(jsd_val))
            if jsd_list:
                results["jsd_mean"] = float(np.mean(jsd_list))

        # Overall count
        if missing_mask.any():
            results["n_missing_values"] = int(missing_mask.sum())

        return results
