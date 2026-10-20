"""
tests/test_data_loading.py
--------------------------
Tests for data_loader.py using real raw files.
These tests verify that the files can be loaded and have the
expected columns and non-trivial row counts.
"""

import os
import pytest
import pandas as pd

RAW_DIR = "data/raw"


def _skip_if_missing(filename):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"Raw file not found: {path}")


class TestLoadCases:
    def test_cases_loads(self):
        _skip_if_missing("covid_19_india.csv")
        from src.data_loader import load_cases
        df = load_cases(RAW_DIR)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 1000

    def test_cases_required_columns(self):
        _skip_if_missing("covid_19_india.csv")
        from src.data_loader import load_cases
        df = load_cases(RAW_DIR)
        for col in ["Date", "State/UnionTerritory", "Confirmed", "Cured", "Deaths"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_cases_date_is_datetime(self):
        _skip_if_missing("covid_19_india.csv")
        from src.data_loader import load_cases
        df = load_cases(RAW_DIR)
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])

    def test_cases_date_range(self):
        _skip_if_missing("covid_19_india.csv")
        from src.data_loader import load_cases
        df = load_cases(RAW_DIR)
        assert df["Date"].min() >= pd.Timestamp("2020-01-01")
        assert df["Date"].max() <= pd.Timestamp("2022-12-31")


class TestLoadTesting:
    def test_testing_loads(self):
        _skip_if_missing("StatewiseTestingDetails.csv")
        from src.data_loader import load_testing
        df = load_testing(RAW_DIR)
        assert len(df) > 100

    def test_testing_columns(self):
        _skip_if_missing("StatewiseTestingDetails.csv")
        from src.data_loader import load_testing
        df = load_testing(RAW_DIR)
        for col in ["Date", "State", "TotalSamples"]:
            assert col in df.columns


class TestLoadVaccine:
    def test_vaccine_loads(self):
        _skip_if_missing("covid_vaccine_statewise.csv")
        from src.data_loader import load_vaccine
        df = load_vaccine(RAW_DIR)
        assert len(df) > 100

    def test_vaccine_date_column(self):
        _skip_if_missing("covid_vaccine_statewise.csv")
        from src.data_loader import load_vaccine
        df = load_vaccine(RAW_DIR)
        assert "Updated On" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Updated On"])


class TestMissingFile:
    def test_missing_cases_raises(self, tmp_path):
        from src.data_loader import load_cases
        with pytest.raises(FileNotFoundError):
            load_cases(str(tmp_path))
