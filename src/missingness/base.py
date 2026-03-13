"""Base missingness generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.registry import MISSINGNESS_REGISTRY


@dataclass
class MissingnessResult:
    """Container for missingness injection outputs."""

    X_missing: pd.DataFrame
    missing_mask: np.ndarray  # True where value is missing
    observed_mask: np.ndarray  # True where value is observed
    missing_config: dict[str, Any] = field(default_factory=dict)
    scenario_tag: str = ""


class BaseMissingnessGenerator(ABC):
    """Abstract base class for missingness generators.

    All generators must produce the same output format so that
    imputation methods receive consistent inputs.
    """

    name: str = ""

    @abstractmethod
    def generate(
        self,
        X: pd.DataFrame,
        metadata: dict[str, Any],
        seed: int = 42,
        rate: float = 0.1,
        **kwargs: Any,
    ) -> MissingnessResult:
        """Inject missing values into X and return MissingnessResult.

        Args:
            X: Complete data (no missing values).
            metadata: Dataset metadata including feature types.
            seed: Random seed for reproducibility.
            rate: Overall missing rate.

        Returns:
            MissingnessResult containing missing data, masks, and config.
        """
