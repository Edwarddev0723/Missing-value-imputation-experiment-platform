#!/usr/bin/env python3
"""Run a single imputation experiment from a config file.

Usage:
    python scripts/run_experiment.py --config configs/experiments/benchmark_core.yaml
    python scripts/run_experiment.py --config configs/experiments/benchmark_core.yaml --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_experiment_config, validate_config
from src.factory import ComponentFactory
from src.orchestration.runner import ExperimentRunner
from src.utils.logging import setup_logging

logger = setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single imputation experiment.")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the experiment config YAML file.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument(
        "--config-dir", type=str, default="configs",
        help="Base config directory.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable W&B tracking (use local tracker).",
    )
    args = parser.parse_args()

    # Load config
    config = load_experiment_config(args.config, config_dir=args.config_dir)

    # Apply overrides
    if args.seed is not None:
        config.setdefault("project", {})["seed"] = args.seed
    if args.no_wandb:
        config.setdefault("tracking", {})["use_wandb"] = False

    # Validate
    errors = validate_config(config)
    if errors:
        for e in errors:
            logger.error(f"Config error: {e}")
        sys.exit(1)

    # Create components
    factory = ComponentFactory()
    dataset = factory.create_dataset(config["dataset"])
    missingness = factory.create_missingness(config["missingness"])
    imputer = factory.create_imputer(config["method"])
    tracker = factory.create_tracker(config.get("tracking", {}))

    # Run
    runner = ExperimentRunner(
        dataset=dataset,
        missingness_generator=missingness,
        imputer=imputer,
        tracker=tracker,
        config=config,
    )
    result = runner.run()

    # Print summary
    logger.info(f"Status: {result.status}")
    if result.imputation_metrics:
        logger.info(f"Imputation metrics: {result.imputation_metrics}")
    if result.downstream_metrics:
        logger.info(f"Downstream metrics: {result.downstream_metrics}")
    if result.runtime_metrics:
        logger.info(f"Runtime: {result.runtime_metrics}")


if __name__ == "__main__":
    main()
