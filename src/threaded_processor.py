"""
threaded_processor.py
---------------------
Laboratory Concept 2: Python Multithreading
============================================

What is multithreading?
-----------------------
Multithreading allows a Python program to run multiple threads of
execution concurrently within a single process, sharing the same memory
space.  Python's Global Interpreter Lock (GIL) means that only one
thread can execute Python bytecode at a time, so multithreading does NOT
provide CPU-bound parallelism for pure Python / NumPy code.

Why use multithreading here?
----------------------------
The analytical tasks below are embarrassingly independent: computing
rolling averages and anomalies for Tamil Nadu is completely independent
of doing the same for Maharashtra, which is independent of Delhi, etc.
Although these are CPU-bound in nature (limited by the GIL for pandas
operations), splitting the work by state:

  (a) demonstrates the pattern clearly for an academic audience,
  (b) can provide wall-clock speedups because pandas releases the GIL
      during some of its underlying NumPy operations,
  (c) is genuinely useful for I/O-bound extensions (reading per-state
      files from disk or network) that students may add later.

We are transparent that GIL-bound speedup is not guaranteed; the actual
measured times are reported in the benchmark section.

Implementation
--------------
We use concurrent.futures.ThreadPoolExecutor, the recommended high-level
threading API in Python 3.  Each state's analytical task (rolling average
+ anomaly detection + weekly summary) is submitted as a separate Future.
Results are collected with executor.map(), which preserves submission
order and re-raises exceptions from worker threads.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data_loader import load_all
from src.data_cleaning import clean_all
from src.analysis import (
    compute_rolling_average,
    weekly_summary,
    detect_anomalies,
    state_comparison_table,
    filter_state,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-state worker function
# ---------------------------------------------------------------------------

def _analyse_one_state(
    cases: pd.DataFrame,
    state: str,
    rolling_window: int = 7,
    z_threshold: float = 2.5,
) -> Dict[str, Any]:
    """
    Perform rolling-average computation, anomaly detection, and weekly
    summary for a single state.  This function is safe to call from
    multiple threads because each call operates on an independent slice
    of the DataFrame (filter_state returns a copy).

    Parameters
    ----------
    cases : pd.DataFrame
        Full cleaned cases DataFrame (read-only in threads).
    state : str
        State name to analyse.
    rolling_window : int
        Days for rolling average.
    z_threshold : float
        Z-score threshold for anomaly detection.

    Returns
    -------
    dict
        Keys: 'state', 'rolling_df', 'anomalies_df', 'weekly_df'
    """
    state_df = filter_state(cases, state)
    if state_df.empty:
        return {"state": state, "rolling_df": pd.DataFrame(),
                "anomalies_df": pd.DataFrame(), "weekly_df": pd.DataFrame()}

    rolling_df = compute_rolling_average(
        state_df, "DailyNewCases", rolling_window, group_col="State"
    )
    rolling_df = compute_rolling_average(
        rolling_df, "DailyNewDeaths", rolling_window, group_col="State"
    )
    anomalies_df = detect_anomalies(
        rolling_df, z_threshold=z_threshold, group_col="State"
    )
    weekly_df = weekly_summary(rolling_df)

    return {
        "state": state,
        "rolling_df": rolling_df,
        "anomalies_df": anomalies_df,
        "weekly_df": weekly_df,
    }


# ---------------------------------------------------------------------------
# Multithreaded runner
# ---------------------------------------------------------------------------

def run_threaded(
    raw_dir: str,
    max_workers: int = 4,
    rolling_window: int = 7,
    z_threshold: float = 2.5,
    states: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the analytical pipeline concurrently using ThreadPoolExecutor.

    Each state's analysis is dispatched as an independent Future.
    The main thread aggregates results once all Futures complete.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw CSV files.
    max_workers : int
        Maximum threads in the pool.
    rolling_window : int
        Days for rolling average.
    z_threshold : float
        Z-score threshold for anomaly detection.
    states : list, optional
        Subset of states to process.  If None, all states are used.

    Returns
    -------
    dict
        Keys: 'results' (list of per-state dicts),
              'combined_rolling' (combined DataFrame),
              'combined_anomalies' (combined DataFrame),
              'state_comparison' (summary table),
              'elapsed_seconds' (float),
              'n_workers' (int),
              'n_states' (int)
    """
    t0 = time.perf_counter()

    # Load and clean (not timed as part of parallel section)
    cases_raw, testing_raw, vaccine_raw = load_all(raw_dir)
    cases, testing, vaccine = clean_all(cases_raw, testing_raw, vaccine_raw)

    all_states = sorted(cases["State"].dropna().unique().tolist())
    if states is not None:
        all_states = [s for s in states if s in all_states]

    logger.info(
        "Launching threaded analysis: %d states, %d workers",
        len(all_states),
        max_workers,
    )

    t_parallel_start = time.perf_counter()
    per_state_results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_state = {
            executor.submit(
                _analyse_one_state, cases, state, rolling_window, z_threshold
            ): state
            for state in all_states
        }
        for future in as_completed(future_to_state):
            state = future_to_state[future]
            try:
                result = future.result()
                per_state_results.append(result)
            except Exception as exc:
                logger.error("State '%s' analysis raised: %s", state, exc)
                raise

    t_parallel_end = time.perf_counter()
    parallel_elapsed = t_parallel_end - t_parallel_start

    # Combine all states
    rolling_frames = [r["rolling_df"] for r in per_state_results if not r["rolling_df"].empty]
    anomaly_frames = [r["anomalies_df"] for r in per_state_results if not r["anomalies_df"].empty]

    combined_rolling = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame()
    combined_anomalies = pd.concat(anomaly_frames, ignore_index=True) if anomaly_frames else pd.DataFrame()

    state_comp = state_comparison_table(cases)

    t1 = time.perf_counter()
    total_elapsed = t1 - t0

    logger.info(
        "Threaded pipeline (parallel section): %.3f s | total: %.3f s",
        parallel_elapsed,
        total_elapsed,
    )

    return {
        "results": per_state_results,
        "combined_rolling": combined_rolling,
        "combined_anomalies": combined_anomalies,
        "state_comparison": state_comp,
        "elapsed_seconds": parallel_elapsed,   # parallel section only for fair comparison
        "total_elapsed_seconds": total_elapsed,
        "n_workers": max_workers,
        "n_states": len(all_states),
    }
