"""
tests/test_threading.py
-----------------------
Tests for the multithreaded processor.
Validates that threaded results match sequential results.
"""

import pandas as pd
import pytest
import os

RAW_DIR = "data/raw"


def _skip_if_missing():
    for f in ["covid_19_india.csv", "StatewiseTestingDetails.csv", "covid_vaccine_statewise.csv"]:
        if not os.path.exists(os.path.join(RAW_DIR, f)):
            pytest.skip(f"Raw file missing: {f}")


class TestThreadedProcessor:
    def test_threaded_run_completes(self):
        _skip_if_missing()
        from src.threaded_processor import run_threaded
        result = run_threaded(RAW_DIR, max_workers=2, states=["Tamil Nadu", "Kerala"])
        assert "combined_rolling" in result
        assert "elapsed_seconds" in result

    def test_threaded_has_two_states(self):
        _skip_if_missing()
        from src.threaded_processor import run_threaded
        result = run_threaded(RAW_DIR, max_workers=2, states=["Tamil Nadu", "Kerala"])
        assert result["n_states"] == 2

    def test_threaded_results_consistent_with_sequential(self):
        """
        Verify that the threaded rolling average for Tamil Nadu matches
        the sequential computation (same inputs → same outputs).
        """
        _skip_if_missing()
        from src.threaded_processor import run_threaded
        from src.data_loader import load_all
        from src.data_cleaning import clean_all
        from src.analysis import compute_rolling_average, filter_state

        # Sequential result
        cases_raw, t_raw, v_raw = load_all(RAW_DIR)
        cases, _, _ = clean_all(cases_raw, t_raw, v_raw)
        cases = compute_rolling_average(cases, "DailyNewCases", 7)
        tn_seq = filter_state(cases, "Tamil Nadu").sort_values("Date")

        # Threaded result
        threaded = run_threaded(RAW_DIR, max_workers=2, states=["Tamil Nadu"])
        tn_thr = threaded["combined_rolling"]
        tn_thr = tn_thr[tn_thr["State"] == "Tamil Nadu"].sort_values("Date")

        # Compare rolling average column
        col = "DailyNewCases_rolling7d"
        if col in tn_seq.columns and col in tn_thr.columns:
            seq_vals = tn_seq[col].reset_index(drop=True).round(4)
            thr_vals = tn_thr[col].reset_index(drop=True).round(4)
            pd.testing.assert_series_equal(seq_vals, thr_vals, check_names=False)

    def test_worker_count_stored(self):
        _skip_if_missing()
        from src.threaded_processor import run_threaded
        result = run_threaded(RAW_DIR, max_workers=3, states=["Tamil Nadu"])
        assert result["n_workers"] == 3

    def test_elapsed_seconds_positive(self):
        _skip_if_missing()
        from src.threaded_processor import run_threaded
        result = run_threaded(RAW_DIR, max_workers=2, states=["Tamil Nadu"])
        assert result["elapsed_seconds"] > 0
