"""Configuration loading, merging, and validation."""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf


@lru_cache(maxsize=256)
def _load_yaml_cached(path_str: str) -> dict[str, Any]:
    """Load a YAML file with LRU caching (hashable key = string path)."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return as dict (cached)."""
    return copy.deepcopy(_load_yaml_cached(str(Path(path).resolve())))


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge multiple config dicts. Later configs override earlier ones."""
    merged: dict[str, Any] = {}
    for cfg in configs:
        merged = _deep_merge(merged, cfg)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_experiment_config(
    experiment_path: str | Path,
    config_dir: str | Path = "configs",
) -> dict[str, Any]:
    """Load a full experiment config by resolving references to sub-configs.

    The experiment config can reference dataset/method/missingness configs
    by name; this function resolves and merges them.
    """
    config_dir = Path(config_dir)
    experiment = load_yaml(experiment_path)

    # Load and merge base config if it exists
    base_path = config_dir / "base.yaml"
    base_cfg = load_yaml(base_path) if base_path.exists() else {}

    # Resolve dataset config reference
    if "dataset" in experiment and isinstance(experiment["dataset"], str):
        dataset_path = config_dir / "datasets" / f"{experiment['dataset']}.yaml"
        if dataset_path.exists():
            experiment["dataset"] = load_yaml(dataset_path).get("dataset", {})

    # Resolve method config reference
    if "method" in experiment and isinstance(experiment["method"], str):
        method_path = config_dir / "methods" / f"{experiment['method']}.yaml"
        if method_path.exists():
            experiment["method"] = load_yaml(method_path).get("method", {})

    # Resolve missingness config reference
    if "missingness" in experiment and isinstance(experiment["missingness"], str):
        miss_path = config_dir / "missingness" / f"{experiment['missingness']}.yaml"
        if miss_path.exists():
            experiment["missingness"] = load_yaml(miss_path).get("missingness", {})

    # Resolve downstream config reference
    if "downstream" in experiment and isinstance(experiment["downstream"], str):
        ds_path = config_dir / "downstream" / f"{experiment['downstream']}.yaml"
        if ds_path.exists():
            experiment["downstream"] = load_yaml(ds_path).get("downstream", {})

    return merge_configs(base_cfg, experiment)


def to_omega_conf(config: dict[str, Any]) -> DictConfig:
    """Convert a plain dict config to OmegaConf DictConfig."""
    return OmegaConf.create(config)


def validate_config(config: dict[str, Any]) -> list[str]:
    """Validate a config dict and return a list of warnings/errors.

    Returns an empty list if the config is valid.
    """
    errors: list[str] = []

    # Check required top-level keys for a single experiment
    if "dataset" not in config and "batch" not in config:
        errors.append("Config must contain either 'dataset' or 'batch' key.")

    if "method" not in config and "batch" not in config:
        errors.append("Config must contain either 'method' or 'batch' key.")

    if "missingness" not in config and "batch" not in config:
        errors.append("Config must contain either 'missingness' or 'batch' key.")

    # Validate dataset section
    if "dataset" in config and isinstance(config["dataset"], dict):
        ds = config["dataset"]
        if "name" not in ds:
            errors.append("dataset.name is required.")

    # Validate method section
    if "method" in config and isinstance(config["method"], dict):
        m = config["method"]
        if "name" not in m:
            errors.append("method.name is required.")

    # Validate missingness section
    if "missingness" in config and isinstance(config["missingness"], dict):
        ms = config["missingness"]
        if "mechanism" not in ms:
            errors.append("missingness.mechanism is required.")

    return errors


def expand_batch_config(
    batch_config: dict[str, Any],
    config_dir: str | Path = "configs",
) -> list[dict[str, Any]]:
    """Expand a batch config into a list of individual experiment configs.

    Generates the cartesian product of datasets × missingness_profiles × methods × seeds.
    Each missingness profile YAML already encodes mechanism + pattern + rate.
    """
    config_dir = Path(config_dir)
    batch = batch_config.get("batch", {})

    datasets = batch.get("datasets", [])
    missingness_profiles = batch.get("missingness_profiles", [])
    methods = batch.get("methods", [])
    seeds = batch.get("seeds", [42])

    # Load base config
    base_path = config_dir / "base.yaml"
    base_cfg = load_yaml(base_path) if base_path.exists() else {}
    # Merge non-batch keys from batch_config
    for key in batch_config:
        if key != "batch":
            base_cfg = merge_configs(base_cfg, {key: batch_config[key]})

    # Pre-load all unique configs once (avoid redundant YAML reads)
    ds_cfgs = {}
    for ds_name in datasets:
        ds_path = config_dir / "datasets" / f"{ds_name}.yaml"
        ds_cfgs[ds_name] = load_yaml(ds_path).get("dataset", {"name": ds_name}) if ds_path.exists() else {"name": ds_name}

    miss_cfgs = {}
    for miss_name in missingness_profiles:
        miss_path = config_dir / "missingness" / f"{miss_name}.yaml"
        miss_cfgs[miss_name] = load_yaml(miss_path).get("missingness", {"mechanism": miss_name}) if miss_path.exists() else {"mechanism": miss_name}

    method_cfgs = {}
    for method_name in methods:
        method_path = config_dir / "methods" / f"{method_name}.yaml"
        method_cfgs[method_name] = load_yaml(method_path).get("method", {"name": method_name}) if method_path.exists() else {"name": method_name}

    base_project = base_cfg.get("project", {})

    experiments = []
    for ds_name in datasets:
        ds_cfg = ds_cfgs[ds_name]
        for miss_name in missingness_profiles:
            miss_cfg = miss_cfgs[miss_name]
            for method_name in methods:
                method_cfg = method_cfgs[method_name]
                for seed in seeds:
                    exp_cfg = merge_configs(base_cfg, {
                        "dataset": ds_cfg,
                        "missingness": miss_cfg,
                        "method": method_cfg,
                        "project": {**base_project, "seed": seed},
                    })
                    experiments.append(exp_cfg)

    return experiments
