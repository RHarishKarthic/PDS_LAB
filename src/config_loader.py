"""
config_loader.py
----------------
Loads and validates the project YAML configuration file.
Falls back to sensible defaults if a key is missing.
"""
import os
import yaml
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "cases_file": "covid_19_india.csv",
        "testing_file": "StatewiseTestingDetails.csv",
        "vaccine_file": "covid_vaccine_statewise.csv",
    },
    "analysis": {
        "focus_state": "Tamil Nadu",
        "rolling_window": 7,
        "anomaly_zscore_threshold": 2.5,
        "min_date": None,
        "max_date": None,
    },
    "spark": {
        "app_name": "PublicHealthOutbreakMonitoring",
        "master": "local[*]",
        "driver_memory": "2g",
        "log_level": "WARN",
    },
    "threading": {"max_workers": 4},
    "benchmarking": {
        "runs": 3,
        "output_file": "outputs/benchmarks/benchmark_results.csv",
    },
    "outputs": {
        "charts_dir": "outputs/charts",
        "reports_dir": "outputs/reports",
        "benchmarks_dir": "outputs/benchmarks",
        "cleaned_cases": "data/processed/cleaned_cases.csv",
        "cleaned_testing": "data/processed/cleaned_testing.csv",
        "cleaned_vaccine": "data/processed/cleaned_vaccine.csv",
    },
    "logging": {"level": "INFO", "file": "outputs/reports/pipeline.log"},
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from *config_path*.

    Falls back to DEFAULT_CONFIG for any missing key.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file (relative to the project root or absolute).

    Returns
    -------
    dict
        Merged configuration dictionary.
    """
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh) or {}
        cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
        logger.debug("Configuration loaded from %s", config_path)
    else:
        cfg = DEFAULT_CONFIG.copy()
        logger.warning(
            "Config file %s not found — using defaults.", config_path
        )
    return cfg
