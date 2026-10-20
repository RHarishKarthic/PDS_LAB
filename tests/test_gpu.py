"""
tests/test_gpu.py
-----------------
Unit tests for the GPU-accelerated outbreak processor module.
Tests both CUDA-accelerated execution and graceful fallback.
"""

import numpy as np
import pandas as pd
import pytest

from src.gpu_processor import (
    get_gpu_info,
    get_cpu_info,
    is_gpu_available,
    _build_lag_features,
    train_gpu_outbreak_model,
    compare_cpu_vs_gpu,
)


class TestGpuHardwareDiscovery:
    def test_gpu_info_returns_dict_with_required_keys(self):
        info = get_gpu_info()
        assert isinstance(info, dict)
        for key in ["available", "name", "driver_version", "total_memory_mb", "backend"]:
            assert key in info

    def test_cpu_info_returns_dict_with_required_keys(self):
        info = get_cpu_info()
        assert isinstance(info, dict)
        for key in ["processor", "physical_cores", "logical_threads", "total_ram_gb", "architecture"]:
            assert key in info
        assert info["physical_cores"] >= 1
        assert info["logical_threads"] >= 1

    def test_is_gpu_available_boolean(self):
        val = is_gpu_available()
        assert isinstance(val, bool)


class TestGpuFeatureEngineering:
    def test_lag_features_created(self, cleaned_cases_df):
        df = _build_lag_features(cleaned_cases_df, target_col="DailyNewCases", lags=(1, 2, 3, 7))
        for col in ["lag_1", "lag_2", "lag_3", "lag_7", "rolling_mean_7", "day_of_week", "month"]:
            assert col in df.columns
        # Target column should still exist
        assert "DailyNewCases" in df.columns
        # Should not have any NaNs in generated features
        assert df[["lag_1", "rolling_mean_7"]].isna().sum().sum() == 0


class TestGpuOutbreakForecaster:
    def test_train_model_cpu_fallback(self, cleaned_cases_df):
        """Verify model training functions correctly on CPU device."""
        state = cleaned_cases_df["State"].iloc[0]
        res = train_gpu_outbreak_model(cleaned_cases_df, focus_state=state, forecast_horizon_days=5, device="cpu")
        assert res["device"] == "cpu"
        assert "metrics" in res
        assert "r2" in res["metrics"]
        assert "evaluation_df" in res
        assert "forecast_df" in res
        assert len(res["forecast_df"]) == 5

    def test_train_model_forecast_horizon(self, cleaned_cases_df):
        state = cleaned_cases_df["State"].iloc[0]
        res = train_gpu_outbreak_model(cleaned_cases_df, focus_state=state, forecast_horizon_days=7, device="cpu")
        assert len(res["forecast_df"]) == 7
        assert "ProjectedDailyNewCases" in res["forecast_df"].columns
        assert (res["forecast_df"]["ProjectedDailyNewCases"] >= 0).all()

    def test_gpu_execution_when_available(self, cleaned_cases_df):
        """If NVIDIA GPU is available, verify CUDA execution completes."""
        if not is_gpu_available():
            pytest.skip("NVIDIA CUDA GPU not detected on this machine.")
        state = cleaned_cases_df["State"].iloc[0]
        res = train_gpu_outbreak_model(cleaned_cases_df, focus_state=state, forecast_horizon_days=5, device="cuda")
        assert res["device"] == "cuda"
        assert res["gpu_kernel_time"] > 0

    def test_compare_cpu_vs_gpu_execution(self, cleaned_cases_df):
        """Verify head-to-head comparison produces valid timings and metrics."""
        state = cleaned_cases_df["State"].iloc[0]
        res = compare_cpu_vs_gpu(cleaned_cases_df, focus_state=state, n_estimators=5, max_depth=2)
        assert "timings" in res
        assert "cpu_train_seconds" in res["timings"]
        assert "gpu_train_seconds" in res["timings"]
        assert "speedup_factor" in res["timings"]
        assert "throughput" in res
        assert "parity" in res
        assert "evaluation_df" in res
        assert res["parity"]["max_abs_diff"] >= 0.0
