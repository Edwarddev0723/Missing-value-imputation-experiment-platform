"""Batch experiment runner — expand experiment matrix and run all combinations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import expand_batch_config, load_experiment_config
from src.factory import ComponentFactory
from src.orchestration.runner import ExperimentRunner
from src.schemas.result_schema import ExperimentResult
from src.utils.io import ensure_dir, save_results_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BatchExperimentRunner:
    """Run a batch of experiments from a batch config.

    Expands dataset × missingness × method × seed and runs each sequentially.
    """

    def __init__(self, config_path: str | None = None, config_dir: str = "configs") -> None:
        self.config_path = config_path
        self.config_dir = config_dir
        self.factory = ComponentFactory()

    def run_from_config(self, config_path: str | None = None) -> list[ExperimentResult]:
        """Load batch config, expand matrix, and run all experiments."""
        path = config_path or self.config_path
        if path is None:
            raise ValueError("config_path must be provided.")

        from src.config import load_yaml
        batch_config = load_yaml(path)
        experiment_configs = expand_batch_config(batch_config, self.config_dir)

        logger.info(f"Batch run: {len(experiment_configs)} experiments to execute.")
        return self.run_all(experiment_configs)

    def run_all(self, experiment_configs: list[dict[str, Any]]) -> list[ExperimentResult]:
        """Run a list of experiment configs and collect results."""
        results: list[ExperimentResult] = []

        for i, cfg in enumerate(experiment_configs):
            ds_name = cfg.get("dataset", {}).get("name", "?")
            method_name = cfg.get("method", {}).get("name", "?")
            mechanism = cfg.get("missingness", {}).get("mechanism", "?")
            seed = cfg.get("project", {}).get("seed", 42)

            logger.info(
                f"[{i+1}/{len(experiment_configs)}] "
                f"dataset={ds_name} method={method_name} "
                f"mechanism={mechanism} seed={seed}"
            )

            try:
                dataset = self.factory.create_dataset(cfg["dataset"])
                missingness = self.factory.create_missingness(cfg["missingness"])
                imputer = self.factory.create_imputer(cfg["method"])
                tracker = self.factory.create_tracker(cfg.get("tracking", {}))

                runner = ExperimentRunner(
                    dataset=dataset,
                    missingness_generator=missingness,
                    imputer=imputer,
                    tracker=tracker,
                    config=cfg,
                )
                result = runner.run()
                results.append(result)

            except Exception as e:
                logger.error(f"Experiment failed: {e}", exc_info=True)
                results.append(ExperimentResult(
                    experiment_id=f"failed_{i}",
                    dataset_name=ds_name,
                    method_name=method_name,
                    method_family="",
                    missing_mechanism=mechanism,
                    missing_rate=cfg.get("missingness", {}).get("rate", 0),
                    scenario_tag="",
                    seed=seed,
                    status="failed",
                    error_message=str(e),
                ))

        # Save summary CSV
        self._save_summary(results)
        return results

    def _save_summary(self, results: list[ExperimentResult]) -> None:
        """Save a summary CSV of all batch results."""
        output_dir = ensure_dir("results/tables")
        flat = [r.flat_dict() for r in results]
        save_results_csv(flat, output_dir / "batch_summary.csv")
        logger.info(f"Batch summary saved to results/tables/batch_summary.csv")
