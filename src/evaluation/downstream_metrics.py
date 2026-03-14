"""Downstream task evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
    """Evaluate how imputation quality affects a fixed downstream model.

    Uses RandomForest as the unified downstream model per experiment spec.
    """

    def fit_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task_type: str = CLASSIFICATION,
        model_params: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> dict[str, float]:
        """Train a fixed downstream model and evaluate on test set."""
        model_params = model_params or {}
        results: dict[str, float] = {}

        try:
            if task_type == CLASSIFICATION:
                model = RandomForestClassifier(
                    n_estimators=model_params.get("n_estimators", 100),
                    random_state=model_params.get("random_state", seed),
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                results["downstream_accuracy"] = float(accuracy_score(y_test, y_pred))
                results["downstream_f1_weighted"] = float(
                    f1_score(y_test, y_pred, average="weighted", zero_division=0)
                )
                results["downstream_f1_macro"] = float(
                    f1_score(y_test, y_pred, average="macro", zero_division=0)
                )

                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        results["downstream_roc_auc"] = float(roc_auc_score(y_test, y_proba))
                    except Exception:
                        pass

            elif task_type == REGRESSION:
                model = RandomForestRegressor(
                    n_estimators=model_params.get("n_estimators", 100),
                    random_state=model_params.get("random_state", seed),
                    n_jobs=-1,
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
