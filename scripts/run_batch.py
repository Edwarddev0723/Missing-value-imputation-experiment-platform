#!/usr/bin/env python3
"""Run a batch of experiments from a batch config file.

Usage:
    python scripts/run_batch.py --config configs/experiments/benchmark_core.yaml
    python scripts/run_batch.py --config configs/experiments/benchmark_core.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Prevent OpenMP (loaded by xgboost) from conflicting with PyTorch MPS on Apple Silicon
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Only set up path — defer ALL heavy imports to main() for spawn-safety
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of imputation experiments.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the batch experiment config YAML file.",
    )
    parser.add_argument(
        "--config-dir", type=str, default="configs",
        help="Base config directory.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B tracking.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes. Default 1 (sequential). "
             "Recommended: 8-10 for non-DL methods, 2-3 for DL methods.",
    )
    args = parser.parse_args()

    # If --no-wandb, set env before import
    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    # Heavy imports INSIDE main() so spawn workers don't trigger them
    from src.orchestration.batch_runner import BatchExperimentRunner
    from src.utils.logging import setup_logging

    logger = setup_logging()

    runner = BatchExperimentRunner(
        config_path=args.config,
        config_dir=args.config_dir,
    )
    results = runner.run_from_config(max_workers=args.workers)

    # Summary
    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")
    logger.info(f"Batch complete: {completed} succeeded, {failed} failed out of {len(results)}.")


if __name__ == "__main__":
    main()
