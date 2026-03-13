"""Downstream task evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from src.constants import CLASSIFICATION, REGRESSION
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DownstreamEvaluator:
    """Evaluate how imputation quality affects a fixed downstream model."""

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = CLASSIFICATION,
        model_params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Train a fixed downstream model and evaluate on test set.

        Args:
            X_train: Imputed training features.
            y_train: Training labels.
            X_test: Imputed test features.
            y_test: Test labels.
            task_type: 'classification' or 'regression'.
            model_params: Optional params for the downstream model.

        Returns:
            Dict of downstream metric values.
        """
        model_params = model_params or {}
        results: dict[str, float] = {}

        try:
            if task_type == CLASSIFICATION:
                model = GradientBoostingClassifier(
                    n_estimators=model_params.get("n_estimators", 100),
                    max_depth=model_params.get("max_depth", 5),
                    learning_rate=model_params.get("learning_rate", 0.1),
                    random_state=model_params.get("random_state", 42),
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results["downstream_accuracy"] = float(accuracy_score(y_test, y_pred))
                results["downstream_f1"] = float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                )
                # ROC-AUC for binary classification
                if len(np.unique(y_test)) == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        results["downstream_roc_auc"] = float(roc_auc_score(y_test, y_proba))
                    except Exception:
                        pass

            elif task_type == REGRESSION:
                model = GradientBoostingRegressor(
                    n_estimators=model_params.get("n_estimators", 100),
                    max_depth=model_params.get("max_depth", 5),
                    learning_rate=model_params.get("learning_rate", 0.1),
                    random_state=model_params.get("random_state", 42),
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results["downstream_rmse"] = float(
                    np.sqrt(mean_squared_error(y_test, y_pred))
                )
                results["downstream_mae"] = float(mean_absolute_error(y_test, y_pred))
                results["downstream_r2"] = float(r2_score(y_test, y_pred))

        except Exception as e:
            logger.warning(f"Downstream evaluation failed: {e}")
            results["downstream_error"] = str(e)

        return results
