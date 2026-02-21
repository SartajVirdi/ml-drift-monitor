"""
utils/config_loader.py
----------------------
Loads, validates, and provides typed access to the YAML config file.

Design choice: We use a plain dict internally rather than a dataclass so that
users can add arbitrary keys without breaking the loader. Validation is
explicit and produces helpful error messages rather than cryptic KeyErrors.
"""

import os
from typing import Any

import yaml

# ── Default values applied when keys are missing from user config ─────────────
DEFAULTS = {
    "data": {
        "reference_path": None,
        "production_path": None,
        "label_column":    None,
    },
    "columns": {
        "numerical":   [],
        "categorical": [],
        "drop":        [],
    },
    "model": {
        "path":      None,
        "framework": "auto",
    },
    "thresholds": {
        "ks_statistic":           0.10,
        "psi":                    0.20,
        "chi2_p_value":           0.05,
        "f1_drop":                0.05,
        "drift_feature_fraction": 0.30,
    },
    "psi": {
        "bins": 10,
    },
    "retraining": {
        "enabled":         True,
        "requires_labels": True,
        "save_path":       "models/retrained_model.pkl",
    },
    "output": {
        "report_path":     "data/drift_report.json",
        "log_path":        "logs/drift_monitor.log",
        "retrain_log_path":"logs/retrain_log.csv",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str) -> dict:
    """
    Load YAML config from disk, merge with defaults, and validate.

    Args:
        config_path: Path to a .yaml config file.

    Returns:
        Fully populated config dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If required fields are missing or invalid.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        user_config = yaml.safe_load(f) or {}

    # Merge user config on top of defaults (user wins on conflict)
    config = _deep_merge(DEFAULTS, user_config)

    _validate(config)
    return config


def config_from_dict(d: dict) -> dict:
    """
    Build a config from a plain dict (e.g. from Streamlit sidebar inputs).
    Merges with defaults so callers only need to specify what they change.
    """
    config = _deep_merge(DEFAULTS, d)
    _validate(config)
    return config


def _validate(config: dict) -> None:
    """Raise ValueError with a clear message if required fields are missing."""
    ref = config["data"]["reference_path"]
    prod = config["data"]["production_path"]

    if not ref:
        raise ValueError(
            "reference_path is required. "
            "Set data.reference_path in your config.yaml or sidebar."
        )
    if not prod:
        raise ValueError(
            "production_path is required. "
            "Set data.production_path in your config.yaml or sidebar."
        )

    # Warn (don't crash) if files don't exist yet — they may be uploaded later
    for label, path in [("reference_path", ref), ("production_path", prod)]:
        if path and not os.path.exists(path):
            pass  # Dashboard handles missing files gracefully


def get(config: dict, *keys, default=None) -> Any:
    """Safe nested key access: get(cfg, 'thresholds', 'psi', default=0.2)"""
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
