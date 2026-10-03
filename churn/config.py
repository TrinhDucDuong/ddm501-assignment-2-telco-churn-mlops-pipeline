"""Portable YAML configuration with explicit environment overrides."""

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Resolve relative data/output paths against the configuration file."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    if not 0 < config.get("capacity", 0) <= 1:
        raise ValueError("capacity must be in (0,1]")
    if not isinstance(config.get("seed"), int):
        raise ValueError("seed must be an integer")
    for key, env in [("data_path", "CHURN_DATA_PATH"), ("output_dir", "CHURN_OUTPUT_DIR")]:
        value = Path(os.environ.get(env, config[key]))
        config[key] = str(value if value.is_absolute() else path.parent / value)
    return config
