"""Experiment runner — single experiment execution pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.constants import STATUS_COMPLETED, STATUS_FAILED
from src.datasets.base import BaseDataset
from src.evaluation.downstream_metrics import DownstreamEvaluator
from src.evaluation.imputation_metrics import ImputationEvaluator
from src.methods.base import BaseImputer
from src.missingness.base import BaseMissingnessGenerator
from src.preprocessing.canonical import CanonicalPreprocessor
from src.schemas.result_schema import ExperimentResult
from src.tracking.wandb_logger import ExperimentTracker
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed
from src.utils.timing import Timer

logger = get_logger(__name__)


class ExperimentRunner:
    """Execute a single imputation experiment end-to-end.

    Pipeline:
        1. Load dataset and split
        2. Preprocess (canonical)
        3. Inject missingness
        4. Fit imputer on train, impute train/test
        5. Evaluate imputation quality
        6. Evaluate downstream task
        7. Log everything to tracker
        8. Save results locally
    """

    def __init__(
        self,
        dataset: BaseDataset,
        missingness_generator: BaseMissingnessGenerator,
        imputer: BaseImputer,
        tracker: ExperimentTracker,
        config: dict[str, Any],
    ) -> None:
        self.dataset = dataset
        self.missingness_generator = missingness_generator
        self.imputer = imputer
        self.tracker = tracker
        self.config = config

        self.imputation_evaluator = ImputationEvaluator()
        self.downstream_evaluator = DownstreamEvaluator()

    def run(self) -> ExperimentResult:
        """Execute the full experiment pipeline and return structured result."""
        seed = self.config.get("project", {}).get("seed", 42)
        experiment_id = self._generate_experiment_id()
        run_name = self._generate_run_name(experiment_id)
        output_dir = self.config.get("project", {}).get("output_dir", "results")

        set_global_seed(seed)
        self.imputer.set_seed(seed)

        # Initialize tracker
        self.tracker.start_run(config=self.config, run_name=run_name)

        try:
            # 1. Load dataset
            logger.info(f"Loading dataset: {self.dataset.name}")
            df = self.dataset.load()
            metadata = self.dataset.get_metadata()

            # 2. Get splits
            split_cfg = self.config.get("dataset", {}).get("split", {})
            splits = self.dataset.get_splits(
                seed=seed,
                train_ratio=split_cfg.get("train_ratio", 0.7),
                valid_ratio=split_cfg.get("valid_ratio", 0.1),
            )
            train_df = splits["train"]
            test_df = splits["test"]

            # 3. Preprocess
            preprocessor = CanonicalPreprocessor(
                encode_categoricals=True, scale_numericals=True,
                drop_duplicates=False,
            )
            train_features = self.dataset.get_features(train_df)
            train_target = self.dataset.get_target(train_df)
            test_features = self.dataset.get_features(test_df)
            test_target = self.dataset.get_target(test_df)

            preprocessor.fit(train_features, metadata)
            train_processed = preprocessor.transform(train_features)
            test_processed = preprocessor.transform(test_features)

            # Keep original complete data for evaluation
            X_train_complete = train_processed.copy()
            X_test_complete = test_processed.copy()

            # 4. Inject missingness
            miss_cfg = self.config.get("missingness", {})
            miss_rate = miss_cfg.get("rate", 0.1)
            miss_seed = miss_cfg.get("seed", seed)
            miss_pattern = miss_cfg.get("pattern", "general")

            logger.info(
                f"Injecting missingness: {self.missingness_generator.name} "
                f"pattern={miss_pattern} rate={miss_rate}"
            )
            train_miss_result = self.missingness_generator.generate(
                X_train_complete, metadata, seed=miss_seed, rate=miss_rate,
                exclude_target=miss_cfg.get("exclude_target", True),
                pattern=miss_pattern,
                scenario_tag=miss_cfg.get("scenario_tag", ""),
            )
            test_miss_result = self.missingness_generator.generate(
                X_test_complete, metadata, seed=miss_seed + 1, rate=miss_rate,
                exclude_target=miss_cfg.get("exclude_target", True),
                pattern=miss_pattern,
                scenario_tag=miss_cfg.get("scenario_tag", ""),
            )

            # 5. Fit imputer and impute
            logger.info(f"Running imputer: {self.imputer.name}")
            with Timer() as fit_timer:
                self.imputer.fit(train_miss_result.X_missing, metadata)

            with Timer() as impute_timer:
                X_train_imputed = self.imputer.impute(train_miss_result.X_missing)
                X_test_imputed = self.imputer.impute(test_miss_result.X_missing)

            # 6. Evaluate imputation quality
            logger.info("Evaluating imputation quality...")
            train_imp_metrics = self.imputation_evaluator.evaluate(
                X_train_complete, X_train_imputed,
                train_miss_result.missing_mask, metadata,
            )
            test_imp_metrics = self.imputation_evaluator.evaluate(
                X_test_complete, X_test_imputed,
                test_miss_result.missing_mask, metadata,
            )
            # Use test metrics as primary
            imp_metrics = {f"test_{k}": v for k, v in test_imp_metrics.items()}
            imp_metrics.update({f"train_{k}": v for k, v in train_imp_metrics.items()})

            # 7. Downstream evaluation
            ds_metrics: dict[str, float] = {}
            ds_cfg = self.config.get("downstream", {})
            if ds_cfg.get("enabled", True) and train_target is not None and test_target is not None:
                logger.info("Evaluating downstream task...")
                task_type = metadata.get("task_type", "classification")
                ds_metrics = self.downstream_evaluator.fit_and_evaluate(
                    X_train_imputed, train_target,
                    X_test_imputed, test_target,
                    task_type=task_type,
                    model_params=ds_cfg.get("params", {}),
                    seed=seed,
                )

            # 8. Runtime metrics
            runtime_metrics = {
                "fit_time_seconds": fit_timer.elapsed,
                "impute_time_seconds": impute_timer.elapsed,
                "total_time_seconds": fit_timer.elapsed + impute_timer.elapsed,
            }

            # 9. Build result (internal)
            result = ExperimentResult(
                experiment_id=experiment_id,
                dataset_name=self.dataset.name,
                method_name=self.imputer.name,
                method_family=self.imputer.family,
                missing_mechanism=self.missingness_generator.name,
                missing_rate=miss_rate,
                missing_pattern=miss_pattern,
                scenario_tag=miss_cfg.get("scenario_tag", ""),
                seed=seed,
                imputation_metrics=imp_metrics,
                downstream_metrics=ds_metrics,
                runtime_metrics=runtime_metrics,
                config=self.config,
                status=STATUS_COMPLETED,
            )

            # 10. Build spec-compliant raw JSON
            dataset_cfg = self.config.get("dataset", {})
            method_cfg = self.config.get("method", {})
            raw_json = {
                "dataset_id": dataset_cfg.get("dataset_id", ""),
                "dataset_name": self.dataset.name,
                "method_id": method_cfg.get("method_id", ""),
                "method_name": self.imputer.name,
                "mechanism": self.missingness_generator.name,
                "pattern": miss_pattern,
                "missing_rate": miss_rate,
                "seed": seed,
                "n_samples": int(X_train_complete.shape[0]),
                "n_features": int(X_train_complete.shape[1]),
                "runtime_seconds": runtime_metrics["total_time_seconds"],
                "imputation_quality": {
                    "numerical": {
                        "rmse": test_imp_metrics.get("numerical_rmse"),
                        "mae": test_imp_metrics.get("numerical_mae"),
                    },
                    "categorical": {
                        "accuracy": test_imp_metrics.get("categorical_accuracy"),
                        "macro_f1": test_imp_metrics.get("categorical_macro_f1"),
                    },
                },
                "downstream_performance": {
                    "task_type": metadata.get("task_type", "classification"),
                    "model": "RandomForest",
                    "metrics": {k.replace("downstream_", ""): v for k, v in ds_metrics.items() if not k.endswith("_error")},
                },
                "distribution_preservation": {
                    "wasserstein_distance_mean": test_imp_metrics.get("wasserstein_distance_mean"),
                    "jsd_mean": test_imp_metrics.get("jsd_mean"),
                },
            }

            # 11. Log to tracker
            all_metrics = {**imp_metrics, **ds_metrics, **runtime_metrics}
            self.tracker.log_metrics(all_metrics)
            self.tracker.log_summary(all_metrics)

            # 12. Save spec-compliant JSON to results/raw/
            save_json(raw_json, ensure_dir(f"{output_dir}/raw") / f"{experiment_id}.json")
            # Also save full result to results/runs/ for internal use
            result_dir = ensure_dir(f"{output_dir}/runs/{experiment_id}")
            save_json(result.to_dict(), result_dir / "result.json")

            self.tracker.finish_run(status=STATUS_COMPLETED)
            logger.info(f"Experiment {experiment_id} completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            self.tracker.finish_run(status=STATUS_FAILED)
            return ExperimentResult(
                experiment_id=experiment_id,
                dataset_name=self.dataset.name,
                method_name=self.imputer.name,
                method_family=self.imputer.family,
                missing_mechanism=self.missingness_generator.name,
                missing_rate=self.config.get("missingness", {}).get("rate", 0.0),
                missing_pattern=self.config.get("missingness", {}).get("pattern", "general"),
                seed=seed,
                status=STATUS_FAILED,
                error_message=str(e),
            )

    def _generate_experiment_id(self) -> str:
        """Generate a deterministic experiment ID from config parameters."""
        ds = self.config.get("dataset", {}).get("name", "unknown")
        method = self.config.get("method", {}).get("name", "unknown")
        mechanism = self.config.get("missingness", {}).get("mechanism", "unknown")
        pattern = self.config.get("missingness", {}).get("pattern", "general")
        rate = self.config.get("missingness", {}).get("rate", 0.0)
        seed = self.config.get("project", {}).get("seed", 0)
        # Convert rate to int percentage for readable ID
        rate_pct = int(round(rate * 100))
        return f"{ds}_{method}_{mechanism}_{pattern}_{rate_pct}pct_s{seed}"

    def _generate_run_name(self, experiment_id: str) -> str:
        return experiment_id

    def result_exists(self) -> bool:
        """Check if a result file already exists for this experiment (for checkpoint/resume)."""
        experiment_id = self._generate_experiment_id()
        output_dir = self.config.get("project", {}).get("output_dir", "results")
        result_path = Path(output_dir) / "raw" / f"{experiment_id}.json"
        return result_path.exists()
