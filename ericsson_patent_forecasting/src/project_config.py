"""
project_config.py

Load the small JSON configuration file used across the coursework pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
import json
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "project_config.json"


@lru_cache(maxsize=1)
def load_project_config() -> Dict:
    """Load the JSON project configuration once per process."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def get_core_settings() -> Dict[str, int]:
    """Return the most commonly used scalar configuration values."""
    config = load_project_config()
    return {
        "train_end_year": int(config["train_end_year"]),
        "val_end_year": int(config["val_end_year"]),
        "future_horizon": int(config["future_horizon"]),
        "random_state": int(config["random_state"]),
        "enforce_non_negative_forecasts": bool(config["enforce_non_negative_forecasts"]),
    }


def get_rolling_validation_windows() -> List[Dict[str, int]]:
    """Return rolling-origin validation windows from the config file."""
    config = load_project_config()
    return [
        {
            "train_end_year": int(window["train_end_year"]),
            "val_start_year": int(window["val_start_year"]),
            "val_end_year": int(window["val_end_year"]),
        }
        for window in config["rolling_validation_windows"]
    ]


def build_parameter_grid(grid_name: str) -> List[Dict]:
    """Convert a config grid section into a list of parameter dictionaries."""
    config = load_project_config()
    grid_spec = config.get(grid_name, {})
    if not grid_spec:
        return []

    keys = list(grid_spec.keys())
    values_product = product(*(grid_spec[key] for key in keys))
    return [
        {key: value for key, value in zip(keys, values)}
        for values in values_product
    ]


def get_text_segmentation_grid() -> List[Dict[str, int]]:
    """Return candidate text segmentation settings."""
    config = load_project_config()
    return [
        {
            "min_df": int(option["min_df"]),
            "max_features": int(option["max_features"]),
            "svd_components": int(option["svd_components"]),
        }
        for option in config["text_segmentation_grid"]
    ]


def get_task3_candidate_k_values() -> List[int]:
    """Return the K values to evaluate for Task 3 clustering."""
    config = load_project_config()
    return [int(k) for k in config["task3_candidate_k_values"]]
