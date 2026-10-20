"""
sequential_processor.py
------------------------
Pure-pandas sequential baseline implementation.

This module performs the same analytical tasks as the multithreaded and
Spark implementations so that execution times can be compared fairly.

Why a sequential baseline?
    Speedup = T_sequential / T_parallel
    Without a reliable sequential time, the speedup ratio is meaningless.

Tasks benchmarked
-----------------
1. Load and clean all three datasets from raw CSVs.
2. Compute rolling-average new-cases for every state.
3. Build the state comparison table.
4. Compute weekly summaries for every state.
5. Run anomaly detection for every state.
"""

import logging
import os
import time
from typing import Dict, Any

import pandas as pd

from src.data_loader import load_all
from src.data_cleaning import clean_all
from src.analysis import (
    compute_rolling_average,
    state_comparison_table,
    weekly_summary,
    detect_anomalies,
)

logger = logging.getLogger(__name__)


def run_sequential(raw_dir: str, rolling_window: int = 7) -> Dict[str, Any]:
    """
    Execute the complete analytical pipeline sequentially (single thread).

    Parameters
    ----------
    raw_dir : str
        Path to the directory containing raw CSV files.
    rolling_window : int
        Number of days for rolling average computation.

    Returns
    -------
    dict
        Keys: 'results' (dict of DataFrames), 'elapsed_seconds' (float)
    """
    t0 = time.perf_counter()

    # --- Load ---
    cases_raw, testing_raw, vaccine_raw = load_all(raw_dir)

    # --- Clean ---
    cases, testing, vaccine = clean_all(cases_raw, testing_raw, vaccine_raw)

    # --- Rolling averages (all states) ---
    cases = compute_rolling_average(cases, "DailyNewCases", rolling_window)
    cases = compute_rolling_average(cases, "DailyNewDeaths", rolling_window)

    # --- State comparison ---
    state_comp = state_comparison_table(cases)

    # --- Weekly summaries ---
    weekly = weekly_summary(cases)

    # --- Anomaly detection ---
    anomalies = detect_anomalies(cases)

    t1 = time.perf_counter()
    elapsed = t1 - t0

    logger.info("Sequential pipeline completed in %.3f s", elapsed)

    return {
        "results": {
            "cases": cases,
            "testing": testing,
            "vaccine": vaccine,
            "state_comparison": state_comp,
            "weekly": weekly,
            "anomalies": anomalies,
        },
        "elapsed_seconds": elapsed,
    }
