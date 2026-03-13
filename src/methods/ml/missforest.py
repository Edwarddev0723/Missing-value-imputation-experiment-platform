"""MissForest imputation — iterative random forest imputer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.constants import FAMILY_ML
from src.methods.base import BaseImputer
from src.registry import METHOD_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


@METHOD_REGISTRY.register("missforest")
class MissForestImputer(BaseImputer):
    """MissForest: Iterative imputation using Random Forest estimators.

    Uses sklearn's IterativeImputer with RandomForestRegressor as the estimator.
    For mixed-type data, all features are treated as numerical after encoding.
    """

    name = "missforest"
    family = FAMILY_ML

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        max_depth: int | None = None,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            max_iter=max_iter,
            max_depth=max_depth,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self._imputer: IterativeImputer | None = None
        self._columns: list[str] = []

    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        self._columns = list(X_train.columns)
        estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=self._seed,
        )
        self._imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            random_state=self._seed,
            sample_posterior=False,
        )
        self._imputer.fit(X_train.values)

    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        if self._imputer is None:
            raise RuntimeError("MissForestImputer must be fit before impute.")
        imputed = self._imputer.transform(X_missing.values)
        return pd.DataFrame(imputed, columns=self._columns, index=X_missing.index)
