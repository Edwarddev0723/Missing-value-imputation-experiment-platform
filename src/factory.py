"""Component factory — creates datasets, methods, missingness generators, trackers from config."""

from __future__ import annotations

from typing import Any

from src.registry import (
    DATASET_REGISTRY,
    METHOD_REGISTRY,
    MISSINGNESS_REGISTRY,
    TRACKER_REGISTRY,
)

# Import concrete implementations to trigger registration
import src.datasets.adult  # noqa: F401
import src.datasets.wine  # noqa: F401
import src.datasets.heart  # noqa: F401
import src.missingness.mcar  # noqa: F401
import src.missingness.mar  # noqa: F401
import src.methods.statistical.mean  # noqa: F401
import src.methods.statistical.knn  # noqa: F401
import src.methods.statistical.softimpute  # noqa: F401
import src.methods.ml.missforest  # noqa: F401
import src.tracking.wandb_logger  # noqa: F401


class ComponentFactory:
    """Factory for creating platform components from config dicts."""

    def create_dataset(self, dataset_cfg: dict[str, Any]) -> Any:
        """Create a dataset adapter from config."""
        name = dataset_cfg["name"]
        kwargs = {k: v for k, v in dataset_cfg.items() if k != "name"}
        return DATASET_REGISTRY.create(name, **kwargs)

    def create_missingness(self, missingness_cfg: dict[str, Any]) -> Any:
        """Create a missingness generator from config."""
        mechanism = missingness_cfg["mechanism"]
        return MISSINGNESS_REGISTRY.create(mechanism)

    def create_imputer(self, method_cfg: dict[str, Any]) -> Any:
        """Create an imputer from config."""
        name = method_cfg["name"]
        params = method_cfg.get("params", {})
        return METHOD_REGISTRY.create(name, **params)

    def create_tracker(self, tracking_cfg: dict[str, Any]) -> Any:
        """Create an experiment tracker from config."""
        use_wandb = tracking_cfg.get("use_wandb", False)
        if use_wandb:
            return TRACKER_REGISTRY.create(
                "wandb",
                project=tracking_cfg.get("project", "missing-value-imputation-benchmark"),
                entity=tracking_cfg.get("entity"),
                group=tracking_cfg.get("group"),
                tags=tracking_cfg.get("tags", []),
                log_artifacts=tracking_cfg.get("log_artifacts", True),
            )
        return TRACKER_REGISTRY.create(
            "local",
            output_dir=tracking_cfg.get("output_dir", "results/logs"),
        )
