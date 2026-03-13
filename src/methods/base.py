"""Base imputer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from src.registry import METHOD_REGISTRY


class BaseImputer(ABC):
    """Abstract base class for all imputation methods.

    Every imputer must implement at minimum:
        - fit(X_train, metadata): Learn from training data.
        - impute(X_missing): Fill missing values and return complete data.

    Optionally override:
        - fit_impute(): Combined fit + impute for methods that require it.
        - get_params(): Return method parameters for logging.
    """

    name: str = ""
    family: str = ""  # 'statistical', 'ml', 'dl'

    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs
        self._seed: int = 42

    def set_seed(self, seed: int) -> None:
        """Set the random seed for this imputer."""
        self._seed = seed

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, metadata: dict[str, Any] | None = None) -> None:
        """Fit the imputer on training data (which may contain missing values)."""

    @abstractmethod
    def impute(self, X_missing: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in X_missing and return a complete DataFrame."""

    def fit_impute(
        self,
        X_train_missing: pd.DataFrame,
        X_eval_missing: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Combined fit and impute.

        Default implementation calls fit() then impute() sequentially.
        Methods that need a joint procedure (e.g., some DL methods) should override.

        Returns:
            If X_eval_missing is None: imputed X_train.
            If X_eval_missing is provided: (imputed_train, imputed_eval).
        """
        self.fit(X_train_missing, metadata)
        imputed_train = self.impute(X_train_missing)
        if X_eval_missing is not None:
            imputed_eval = self.impute(X_eval_missing)
            return imputed_train, imputed_eval
        return imputed_train

    def get_params(self) -> dict[str, Any]:
        """Return method parameters for logging/tracking."""
        return {
            "name": self.name,
            "family": self.family,
            "seed": self._seed,
            **self.params,
        }
