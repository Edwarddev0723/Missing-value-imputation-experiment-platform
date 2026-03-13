"""Base experiment tracker and W&B tracker implementation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.registry import TRACKER_REGISTRY
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""

    @abstractmethod
    def start_run(self, config: dict[str, Any], **kwargs: Any) -> None:
        """Initialize a new run."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metric values."""

    @abstractmethod
    def log_artifact(self, path: str, artifact_type: str = "result") -> None:
        """Log a file/directory as an artifact."""

    @abstractmethod
    def log_summary(self, summary: dict[str, Any]) -> None:
        """Log summary metrics at the end of a run."""

    @abstractmethod
    def finish_run(self, status: str = "completed") -> None:
        """Finalize the current run."""


@TRACKER_REGISTRY.register("wandb")
class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker."""

    def __init__(
        self,
        project: str = "missing-value-imputation-benchmark",
        entity: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        log_artifacts: bool = True,
        **kwargs: Any,
    ) -> None:
        self.project = project
        self.entity = entity
        self.group = group
        self.tags = tags or []
        self.log_artifacts_flag = log_artifacts
        self._run = None

    def start_run(self, config: dict[str, Any], **kwargs: Any) -> None:
        import wandb

        run_name = kwargs.get("run_name")
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            group=self.group,
            tags=self.tags,
            config=config,
            name=run_name,
            reinit=True,
        )
        logger.info(f"W&B run started: {self._run.name} ({self._run.id})")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        import wandb

        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def log_artifact(self, path: str, artifact_type: str = "result") -> None:
        if self._run is None or not self.log_artifacts_flag:
            return
        import wandb

        artifact_name = Path(path).stem
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        p = Path(path)
        if p.is_dir():
            artifact.add_dir(str(p))
        elif p.is_file():
            artifact.add_file(str(p))
        self._run.log_artifact(artifact)
        logger.info(f"Logged artifact: {artifact_name}")

    def log_summary(self, summary: dict[str, Any]) -> None:
        if self._run is None:
            return
        for key, value in summary.items():
            self._run.summary[key] = value

    def finish_run(self, status: str = "completed") -> None:
        if self._run is None:
            return
        import wandb

        self._run.summary["status"] = status
        self._run.finish()
        logger.info(f"W&B run finished with status: {status}")
        self._run = None


@TRACKER_REGISTRY.register("local")
class LocalTracker(ExperimentTracker):
    """Local file-based tracker — saves metrics and config as JSON files."""

    def __init__(self, output_dir: str = "results/logs", **kwargs: Any) -> None:
        self.output_dir = Path(output_dir)
        self._run_config: dict[str, Any] = {}
        self._metrics: list[dict[str, Any]] = []
        self._run_name: str = "unnamed"

    def start_run(self, config: dict[str, Any], **kwargs: Any) -> None:
        self._run_config = config
        self._metrics = []
        self._run_name = kwargs.get("run_name", "unnamed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Local tracker started: {self._run_name}")

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        entry = {**metrics}
        if step is not None:
            entry["_step"] = step
        self._metrics.append(entry)

    def log_artifact(self, path: str, artifact_type: str = "result") -> None:
        # Local tracker just logs the path
        logger.info(f"Local artifact recorded: {path} (type={artifact_type})")

    def log_summary(self, summary: dict[str, Any]) -> None:
        self._run_config["summary"] = summary

    def finish_run(self, status: str = "completed") -> None:
        run_dir = self.output_dir / self._run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._run_config, f, indent=2, default=str)

        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self._metrics, f, indent=2, default=str)

        self._run_config["status"] = status
        logger.info(f"Local tracker finished: {self._run_name} ({status})")
