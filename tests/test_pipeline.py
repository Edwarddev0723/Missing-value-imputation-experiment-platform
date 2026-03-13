"""End-to-end pipeline smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.datasets.base import BaseDataset
from src.evaluation.imputation_metrics import ImputationEvaluator
from src.evaluation.downstream_metrics import DownstreamEvaluator
from src.methods.statistical.mean import MeanImputer
from src.methods.statistical.knn import KNNImputer
from src.missingness.mcar import MCARGenerator
from src.orchestration.runner import ExperimentRunner
from src.tracking.wandb_logger import LocalTracker
from src.registry import DATASET_REGISTRY

import src.datasets.adult  # noqa: F401


class InMemoryDataset(BaseDataset):
    """Tiny in-memory dataset for pipeline testing."""

    name = "test_inmemory"

    def __init__(self) -> None:
        np.random.seed(42)
        n = 200
        self._df = pd.DataFrame({
            "feat_a": np.random.randn(n),
            "feat_b": np.random.randn(n) * 2 + 5,
            "feat_c": np.random.rand(n) * 10,
            "target": np.random.randint(0, 2, n),
        })

    def load(self) -> pd.DataFrame:
        return self._df.copy()

    def get_metadata(self) -> dict:
        return {
            "dataset_name": "test_inmemory",
            "n_samples": len(self._df),
            "n_features": 3,
            "task_type": "classification",
            "target_col": "target",
            "numerical_features": ["feat_a", "feat_b", "feat_c"],
            "categorical_features": [],
            "feature_types": {
                "feat_a": "numerical",
                "feat_b": "numerical",
                "feat_c": "numerical",
            },
        }

    def get_splits(self, seed=42, train_ratio=0.7, valid_ratio=0.1):
        from src.datasets.utils import stratified_split
        return stratified_split(self._df, "target", train_ratio, valid_ratio, seed)


class TestEndToEndPipeline:
    """Smoke test: full pipeline with in-memory data."""

    def test_mean_pipeline(self, tmp_path):
        config = {
            "project": {"seed": 42, "output_dir": str(tmp_path)},
            "dataset": {"name": "test_inmemory"},
            "missingness": {"mechanism": "MCAR", "rate": 0.2, "seed": 42},
            "method": {"name": "mean"},
            "tracking": {"use_wandb": False},
            "downstream": {"enabled": True},
        }

        runner = ExperimentRunner(
            dataset=InMemoryDataset(),
            missingness_generator=MCARGenerator(),
            imputer=MeanImputer(),
            tracker=LocalTracker(output_dir=str(tmp_path / "logs")),
            config=config,
        )
        result = runner.run()

        assert result.status == "completed"
        assert result.dataset_name == "test_inmemory"
        assert result.method_name == "mean"
        assert "test_numerical_rmse" in result.imputation_metrics
        assert "downstream_accuracy" in result.downstream_metrics
        assert result.runtime_metrics["total_time_seconds"] > 0

    def test_knn_pipeline(self, tmp_path):
        config = {
            "project": {"seed": 42, "output_dir": str(tmp_path)},
            "dataset": {"name": "test_inmemory"},
            "missingness": {"mechanism": "MCAR", "rate": 0.15, "seed": 42},
            "method": {"name": "knn"},
            "tracking": {"use_wandb": False},
            "downstream": {"enabled": True},
        }

        runner = ExperimentRunner(
            dataset=InMemoryDataset(),
            missingness_generator=MCARGenerator(),
            imputer=KNNImputer(n_neighbors=3),
            tracker=LocalTracker(output_dir=str(tmp_path / "logs")),
            config=config,
        )
        result = runner.run()

        assert result.status == "completed"
        assert result.method_name == "knn"

    def test_result_saved_locally(self, tmp_path):
        config = {
            "project": {"seed": 42, "output_dir": str(tmp_path)},
            "dataset": {"name": "test_inmemory"},
            "missingness": {"mechanism": "MCAR", "rate": 0.1, "seed": 42},
            "method": {"name": "mean"},
            "tracking": {"use_wandb": False},
            "downstream": {"enabled": False},
        }

        runner = ExperimentRunner(
            dataset=InMemoryDataset(),
            missingness_generator=MCARGenerator(),
            imputer=MeanImputer(),
            tracker=LocalTracker(output_dir=str(tmp_path / "logs")),
            config=config,
        )
        result = runner.run()

        # Check that result JSON was saved
        runs_dir = tmp_path / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) > 0

    def test_flat_dict_for_csv(self, tmp_path):
        config = {
            "project": {"seed": 42, "output_dir": str(tmp_path)},
            "dataset": {"name": "test_inmemory"},
            "missingness": {"mechanism": "MCAR", "rate": 0.1, "seed": 42},
            "method": {"name": "mean"},
            "tracking": {"use_wandb": False},
            "downstream": {"enabled": True},
        }

        runner = ExperimentRunner(
            dataset=InMemoryDataset(),
            missingness_generator=MCARGenerator(),
            imputer=MeanImputer(),
            tracker=LocalTracker(output_dir=str(tmp_path / "logs")),
            config=config,
        )
        result = runner.run()
        flat = result.flat_dict()

        assert "experiment_id" in flat
        assert "method_name" in flat
        assert "dataset_name" in flat
