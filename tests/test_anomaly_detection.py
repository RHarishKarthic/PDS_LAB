"""
tests/test_anomaly_detection.py
--------------------------------
Tests for the standalone anomaly_detection module.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def spike_df():
    """Synthetic DataFrame with one clear spike anomaly."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    values = [10] * 29 + [500]    # sudden spike on day 30
    return pd.DataFrame({
        "Date": dates,
        "State": "TestState",
        "DailyNewCases": values,
    })


class TestRollingZscoreAnomalies:
    def test_spike_detected(self, spike_df):
        from src.anomaly_detection import rolling_zscore_anomalies
        result = rolling_zscore_anomalies(spike_df, z_threshold=2.5)
        # The spike on the last day should be flagged
        assert result.iloc[-1]["IsAnomaly"] is True or result.iloc[-1]["IsAnomaly"] == True

    def test_flat_series_no_anomalies(self):
        """A perfectly flat series has ZScore=0 everywhere; no anomalies."""
        from src.anomaly_detection import rolling_zscore_anomalies
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        df = pd.DataFrame({
            "Date": dates,
            "State": "TestState",
            "DailyNewCases": [100] * 20,
        })
        result = rolling_zscore_anomalies(df, z_threshold=2.5)
        assert result["IsAnomaly"].sum() == 0

    def test_required_columns_present(self, spike_df):
        from src.anomaly_detection import rolling_zscore_anomalies
        result = rolling_zscore_anomalies(spike_df)
        for col in ["ZScore", "IsAnomaly", "AnomalyDirection"]:
            assert col in result.columns

    def test_anomaly_direction_values(self, spike_df):
        from src.anomaly_detection import rolling_zscore_anomalies
        result = rolling_zscore_anomalies(spike_df)
        valid = {"Spike", "Drop", "Normal"}
        assert set(result["AnomalyDirection"].unique()).issubset(valid)

    def test_is_anomaly_bool(self, spike_df):
        from src.anomaly_detection import rolling_zscore_anomalies
        result = rolling_zscore_anomalies(spike_df)
        assert result["IsAnomaly"].dtype == bool

    def test_row_count_preserved(self, spike_df):
        from src.anomaly_detection import rolling_zscore_anomalies
        result = rolling_zscore_anomalies(spike_df)
        assert len(result) == len(spike_df)


class TestAnomalySummary:
    def test_summary_count(self, spike_df):
        from src.anomaly_detection import rolling_zscore_anomalies, anomaly_summary
        result = rolling_zscore_anomalies(spike_df, z_threshold=2.5)
        summary = anomaly_summary(result)
        # The spike should be in the summary
        assert len(summary) > 0
        assert "AnomalyCount" in summary.columns

    def test_empty_anomalies_returns_empty_summary(self):
        from src.anomaly_detection import anomaly_summary
        df = pd.DataFrame({
            "State": [], "IsAnomaly": [], "Date": [], "ZScore": []
        })
        df["IsAnomaly"] = df["IsAnomaly"].astype(bool)
        result = anomaly_summary(df)
        assert result.empty
