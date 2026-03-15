"""Batch experiment runner — expand experiment matrix and run all combinations."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import expand_batch_config, load_experiment_config
from src.factory import ComponentFactory
from src.orchestration.runner import ExperimentRunner
from src.schemas.result_schema import ExperimentResult
from src.utils.io import ensure_dir, save_results_csv
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Estimated cost model for smart scheduling ─────────────────────────────────

# Approximate relative cost per method (higher = slower)
_METHOD_COST = {
    "xgboost": 50, "missforest": 30, "mice": 8, "em": 5,
    "softimpute": 3, "knn": 2, "mean": 1, "median": 1,
    # DL methods
    "mida": 10, "miwae": 15, "gain": 5, "tabcsdi": 10, "miracle": 15,
}
# Approximate relative cost per dataset (scaled by size × features)
_DATASET_COST = {
    "spambase": 50, "adult": 30, "letter": 20, "bank": 25,
    "wine": 5, "yeast": 3, "breast_cancer": 2, "concrete": 2,
    "titanic": 2, "mammographic": 2, "autism": 2, "heart": 2,
}


def _sort_by_estimated_cost(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort configs so expensive experiments run first (reduces tail latency)."""
    def _cost(cfg: dict) -> float:
        m = cfg.get("method", {}).get("name", "")
        d = cfg.get("dataset", {}).get("name", "")
        return _METHOD_COST.get(m, 5) * _DATASET_COST.get(d, 5)
    return sorted(configs, key=_cost, reverse=True)


# ── Top-level worker (must be picklable for spawn context) ────────────────────

def _worker_init(max_workers: int = 1) -> None:
    """Initializer for each worker process — runs once at startup."""
    import logging
    import os
    import warnings

    # Suppress ALL warnings at the process level (before any imports)
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"

    # Allow 2 threads per worker for tree-based methods (RF, XGB)
    # that benefit from intra-model parallelism while avoiding over-subscription
    threads = max(2, (os.cpu_count() or 4) // max(max_workers, 1))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "BLIS_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
        os.environ[var] = str(threads)

    # Disable MPS in worker processes — MPS + multiprocessing spawn = segfault
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    logging.disable(logging.WARNING)


def _worker_run_experiment(cfg: dict) -> dict:
    """Execute a single experiment inside a worker process."""
    import os
    import warnings
    warnings.simplefilter("ignore")  # belt-and-suspenders: re-suppress per task

    # Limit n_jobs inside workers to prevent nested multiprocessing deadlock
    # Allow 2 threads for tree-based methods that benefit from parallelism
    n_cpus = os.cpu_count() or 4
    worker_threads = max(2, n_cpus // 8)  # ~2 threads per worker with 8 workers
    if "params" in cfg.get("method", {}):
        cfg["method"]["params"]["n_jobs"] = worker_threads
    os.environ["LOKY_MAX_CPU_COUNT"] = str(worker_threads)

    from src.factory import ComponentFactory
    from src.orchestration.runner import ExperimentRunner

    ds = cfg.get("dataset", {}).get("name", "?")
    method = cfg.get("method", {}).get("name", "?")
    mechanism = cfg.get("missingness", {}).get("mechanism", "?")
    seed = cfg.get("project", {}).get("seed", 42)
    tag = f"{ds}_{method}_{mechanism}_s{seed}"

    factory = ComponentFactory()
    try:
        dataset = factory.create_dataset(cfg["dataset"])
        missingness = factory.create_missingness(cfg["missingness"])
        imputer = factory.create_imputer(cfg["method"])
        tracker = factory.create_tracker(cfg.get("tracking", {}))

        runner = ExperimentRunner(
            dataset=dataset,
            missingness_generator=missingness,
            imputer=imputer,
            tracker=tracker,
            config=cfg,
        )
        if runner.result_exists():
            return {"status": "skipped", "tag": tag}

        result = runner.run()
        if result.status != "completed":
            return {"status": "failed", "tag": tag, "error": result.error_message or "unknown"}
        return {"status": "completed", "tag": tag}
    except Exception as e:
        return {"status": "failed", "tag": tag, "error": str(e)}


class BatchExperimentRunner:
    """Run a batch of experiments from a batch config.

    Expands dataset × missingness × method × seed and runs each sequentially.
    Supports checkpoint/resume: skips runs whose result JSON already exists.
    """

    def __init__(self, config_path: str | None = None, config_dir: str = "configs") -> None:
        self.config_path = config_path
        self.config_dir = config_dir
        self.factory = ComponentFactory()

    def run_from_config(
        self, config_path: str | None = None, max_workers: int = 1,
    ) -> list[ExperimentResult]:
        """Load batch config, expand matrix, and run all experiments."""
        path = config_path or self.config_path
        if path is None:
            raise ValueError("config_path must be provided.")

        from src.config import load_yaml
        batch_config = load_yaml(path)
        experiment_configs = expand_batch_config(batch_config, self.config_dir)

        logger.info(f"Batch run: {len(experiment_configs)} experiments to execute.")
        if max_workers > 1:
            return self.run_all_parallel(experiment_configs, max_workers=max_workers)
        return self.run_all(experiment_configs)

    def run_all(self, experiment_configs: list[dict[str, Any]]) -> list[ExperimentResult]:
        """Run a list of experiment configs and collect results."""
        from tqdm import tqdm

        results: list[ExperimentResult] = []
        skipped = 0
        failed_count = 0

        pbar = tqdm(
            enumerate(experiment_configs),
            total=len(experiment_configs),
            desc="Experiments",
            unit="exp",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )
        for i, cfg in pbar:
            ds_name = cfg.get("dataset", {}).get("name", "?")
            method_name = cfg.get("method", {}).get("name", "?")
            mechanism = cfg.get("missingness", {}).get("mechanism", "?")
            pattern = cfg.get("missingness", {}).get("pattern", "general")
            rate = cfg.get("missingness", {}).get("rate", 0.0)
            seed = cfg.get("project", {}).get("seed", 42)

            pbar.set_postfix_str(
                f"{ds_name}/{method_name}/{mechanism} ok={len(results)} skip={skipped} fail={failed_count}",
                refresh=False,
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

                # Checkpoint/resume: skip if result already exists
                if runner.result_exists():
                    skipped += 1
                    continue

                result = runner.run()
                results.append(result)

            except Exception as e:
                failed_count += 1
                tqdm.write(f"FAIL [{ds_name}/{method_name}/{mechanism}_s{seed}]: {e}")
                results.append(ExperimentResult(
                    experiment_id=f"failed_{i}",
                    dataset_name=ds_name,
                    method_name=method_name,
                    method_family="",
                    missing_mechanism=mechanism,
                    missing_rate=rate,
                    missing_pattern=pattern,
                    seed=seed,
                    status="failed",
                    error_message=str(e),
                ))

        pbar.close()
        logger.info(f"Batch complete: {len(results)} ran, {skipped} skipped, {failed_count} failed")

        # Save summary CSV
        if results:
            self._save_summary(results)
        return results

    # ── Warm cache ─────────────────────────────────────────────────────────

    def _warm_cache(self, experiment_configs: list[dict[str, Any]]) -> None:
        """Pre-download all unique datasets sequentially to avoid race conditions."""
        seen: set[str] = set()
        for cfg in experiment_configs:
            ds_name = cfg.get("dataset", {}).get("name", "")
            if ds_name and ds_name not in seen:
                seen.add(ds_name)
        logger.info(f"Warming cache for {len(seen)} dataset(s): {sorted(seen)}")
        for ds_name in sorted(seen):
            try:
                ds = self.factory.create_dataset({"name": ds_name})
                ds.load()
                logger.info(f"  ✓ {ds_name} cached")
            except Exception as e:
                logger.error(f"  ✗ {ds_name} failed: {e}")

    # ── Parallel execution ────────────────────────────────────────────────

    def run_all_parallel(
        self,
        experiment_configs: list[dict[str, Any]],
        max_workers: int = 4,
    ) -> list[ExperimentResult]:
        """Run experiments in parallel using ProcessPoolExecutor.

        Uses 'spawn' context for MPS compatibility.
        Sorts experiments so slow combos (large datasets × heavy methods) start first,
        avoiding long-tail wait at the end.
        """
        import time

        # Pre-download datasets to avoid race conditions in workers
        self._warm_cache(experiment_configs)

        # Smart scheduling: sort slow experiments first to minimize tail latency
        experiment_configs = _sort_by_estimated_cost(experiment_configs)

        total = len(experiment_configs)
        ctx = mp.get_context("spawn")

        completed = 0
        skipped = 0
        failed = 0
        failed_tags: list[str] = []
        t0 = time.monotonic()

        from tqdm import tqdm

        logger.info(
            f"Starting parallel batch: {total} experiments, {max_workers} workers"
        )

        pbar = tqdm(
            total=total,
            desc=f"Parallel ({max_workers}w)",
            unit="exp",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        )

        from functools import partial

        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx,
            initializer=partial(_worker_init, max_workers),
        ) as executor:
            futures = {
                executor.submit(_worker_run_experiment, cfg): i
                for i, cfg in enumerate(experiment_configs)
            }

            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    failed_tags.append(f"process_error: {exc}")
                    pbar.update(1)
                    continue

                status = result.get("status", "failed")
                tag = result.get("tag", "?")
                if status == "completed":
                    completed += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    failed_tags.append(
                        f"{tag}: {result.get('error', '?')}"
                    )
                    tqdm.write(f"FAIL [{tag}]: {result.get('error', '?')}")

                pbar.update(1)
                pbar.set_postfix_str(
                    f"ok={completed} skip={skipped} fail={failed}",
                    refresh=False,
                )

        pbar.close()
        elapsed = time.monotonic() - t0
        logger.info(
            f"Parallel batch complete in {elapsed:.0f}s: "
            f"{completed} done, {skipped} skipped, {failed} failed"
        )
        if failed_tags:
            for tag in failed_tags[:20]:
                logger.warning(f"  FAILED: {tag}")
            if len(failed_tags) > 20:
                logger.warning(f"  ... and {len(failed_tags) - 20} more failures")

        # Return empty list (results are saved to disk by workers)
        return []

    def _save_summary(self, results: list[ExperimentResult]) -> None:
        """Save a summary CSV of all batch results."""
        output_dir = ensure_dir("results/tables")
        flat = [r.flat_dict() for r in results]
        save_results_csv(flat, output_dir / "batch_summary.csv")
        logger.info(f"Batch summary saved to results/tables/batch_summary.csv")
