"""Result schema for structured experiment outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run."""

    experiment_id: str
    dataset_name: str
    method_name: str
    method_family: str
    missing_mechanism: str
    missing_rate: float
    scenario_tag: str
    seed: int
    imputation_metrics: dict[str, float] = field(default_factory=dict)
    downstream_metrics: dict[str, float] = field(default_factory=dict)
    runtime_metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_dict(self) -> dict[str, Any]:
        """Return a flat dict suitable for CSV/summary table."""
        flat: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "dataset_name": self.dataset_name,
            "method_name": self.method_name,
            "method_family": self.method_family,
            "missing_mechanism": self.missing_mechanism,
            "missing_rate": self.missing_rate,
            "scenario_tag": self.scenario_tag,
            "seed": self.seed,
            "status": self.status,
        }
        for k, v in self.imputation_metrics.items():
            flat[f"imp_{k}"] = v
        for k, v in self.downstream_metrics.items():
            flat[f"ds_{k}"] = v
        for k, v in self.runtime_metrics.items():
            flat[f"rt_{k}"] = v
        return flat
