#!/usr/bin/env python3
"""Run a batch of experiments from a batch config file.

Usage:
    python scripts/run_batch.py --config configs/experiments/benchmark_core.yaml
    python scripts/run_batch.py --config configs/experiments/benchmark_core.yaml --no-wandb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.orchestration.batch_runner import BatchExperimentRunner
from src.utils.logging import setup_logging

logger = setup_logging()


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
    args = parser.parse_args()

    # If --no-wandb, set env before import
    if args.no_wandb:
        import os
        os.environ["WANDB_MODE"] = "disabled"

    runner = BatchExperimentRunner(
        config_path=args.config,
        config_dir=args.config_dir,
    )
    results = runner.run_from_config()

    # Summary
    completed = sum(1 for r in results if r.status == "completed")
    failed = sum(1 for r in results if r.status == "failed")
    logger.info(f"Batch complete: {completed} succeeded, {failed} failed out of {len(results)}.")


if __name__ == "__main__":
    main()
