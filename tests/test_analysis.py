"""
tests/test_analysis.py
-----------------------
Tests for analysis functions using synthetic fixtures.
"""

import numpy as np
import pandas as pd
import pytest


class TestFilterState:
    def test_returns_only_state(self, cleaned_cases_df):
        from src.analysis import filter_state
        tn = filter_state(cleaned_cases_df, "Tamil Nadu")
        assert all(tn["State"] == "Tamil Nadu")
        assert len(tn) > 0

    def test_missing_state_returns_empty(self, cleaned_cases_df):
        from src.analysis import filter_state
        df = filter_state(cleaned_cases_df, "Nonexistent State XYZ")
        assert df.empty

    def test_returns_copy(self, cleaned_cases_df):
        from src.analysis import filter_state
        tn = filter_state(cleaned_cases_df, "Tamil Nadu")
        tn["__test__"] = 1
        assert "__test__" not in cleaned_cases_df.columns


class TestRollingAverage:
    def test_output_column_exists(self, cleaned_cases_df):
        from src.analysis import compute_rolling_average
        result = compute_rolling_average(cleaned_cases_df, "DailyNewCases", 3)
        assert "DailyNewCases_rolling3d" in result.columns

    def test_rolling_values_are_non_negative(self, cleaned_cases_df):
        from src.analysis import compute_rolling_average
        result = compute_rolling_average(cleaned_cases_df, "DailyNewCases", 3)
        assert (result["DailyNewCases_rolling3d"].dropna() >= 0).all()

    def test_rolling_window_of_1_equals_original(self, cleaned_cases_df):
        """Window=1 rolling average should equal the original values."""
        from src.analysis import compute_rolling_average
        result = compute_rolling_average(cleaned_cases_df, "DailyNewCases", 1)
        pd.testing.assert_series_equal(
            result["DailyNewCases"].reset_index(drop=True),
            result["DailyNewCases_rolling1d"].reset_index(drop=True),
            check_names=False,
        )


class TestWeeklySummary:
    def test_returns_dataframe(self, cleaned_cases_df):
        from src.analysis import weekly_summary
        result = weekly_summary(cleaned_cases_df)
        assert isinstance(result, pd.DataFrame)
        assert "YearWeek" in result.columns

    def test_weekly_sum_ge_any_single_day(self, cleaned_cases_df):
        from src.analysis import weekly_summary
        weekly = weekly_summary(cleaned_cases_df)
        if "DailyNewCases" in weekly.columns:
            # Weekly total >= max daily single value would hold for any 7-day window
            max_daily = cleaned_cases_df["DailyNewCases"].max()
            # At least some week total should equal or exceed max_daily (if cases exist)
            assert weekly["DailyNewCases"].max() >= max_daily


class TestMonthlySummary:
    def test_returns_dataframe(self, cleaned_cases_df):
        from src.analysis import monthly_summary
        result = monthly_summary(cleaned_cases_df)
        assert isinstance(result, pd.DataFrame)
        assert "YearMonth" in result.columns
        assert "MonthStart" in result.columns


class TestStateComparisonTable:
    def test_returns_all_states(self, cleaned_cases_df):
        from src.analysis import state_comparison_table
        table = state_comparison_table(cleaned_cases_df)
        states_in_data = cleaned_cases_df["State"].unique()
        # All original states should appear
        for s in states_in_data:
            assert s in table["State"].values

    def test_sorted_by_confirmed(self, cleaned_cases_df):
        from src.analysis import state_comparison_table
        table = state_comparison_table(cleaned_cases_df)
        # TotalConfirmed should be descending
        vals = table["TotalConfirmed"].tolist()
        assert vals == sorted(vals, reverse=True)

    def test_required_columns(self, cleaned_cases_df):
        from src.analysis import state_comparison_table
        table = state_comparison_table(cleaned_cases_df)
        for col in ["State", "TotalConfirmed", "TotalDeaths", "PeakDailyNewCases"]:
            assert col in table.columns


class TestAnomalyDetection:
    def test_returns_required_columns(self, cleaned_cases_df):
        from src.analysis import detect_anomalies
        result = detect_anomalies(cleaned_cases_df)
        for col in ["ZScore", "IsAnomaly"]:
            assert col in result.columns

    def test_is_anomaly_boolean(self, cleaned_cases_df):
        from src.analysis import detect_anomalies
        result = detect_anomalies(cleaned_cases_df)
        assert result["IsAnomaly"].dtype == bool

    def test_zscore_within_reasonable_range(self, cleaned_cases_df):
        from src.analysis import detect_anomalies
        result = detect_anomalies(cleaned_cases_df)
        # Z scores should not be astronomically large
        assert result["ZScore"].abs().max() < 100


class TestTNDetailedAnalysis:
    def test_returns_all_keys(self, cleaned_cases_df, cleaned_testing_df, cleaned_vaccine_df):
        from src.analysis import tn_detailed_analysis
        result = tn_detailed_analysis(cleaned_cases_df, cleaned_testing_df, cleaned_vaccine_df)
        for key in ["cases_tn", "testing_tn", "vaccine_tn", "weekly_tn",
                    "monthly_tn", "anomalies_tn", "state_comparison"]:
            assert key in result

    def test_tn_subset_correct(self, cleaned_cases_df, cleaned_testing_df, cleaned_vaccine_df):
        from src.analysis import tn_detailed_analysis
        result = tn_detailed_analysis(cleaned_cases_df, cleaned_testing_df, cleaned_vaccine_df)
        tn = result["cases_tn"]
        assert all(tn["State"] == "Tamil Nadu")
