"""
tests/test_validation.py
------------------------
Tests for the validation module.
"""

import pandas as pd
import pytest


class TestValidateCases:
    def test_valid_passes(self, cleaned_cases_df):
        from src.validation import validate_cases
        validate_cases(cleaned_cases_df)   # should not raise

    def test_empty_raises(self):
        from src.validation import validate_cases
        with pytest.raises(ValueError, match="empty"):
            validate_cases(pd.DataFrame())

    def test_missing_column_raises(self, cleaned_cases_df):
        from src.validation import validate_cases
        df = cleaned_cases_df.drop(columns=["DailyNewCases"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_cases(df)

    def test_non_datetime_date_raises(self, cleaned_cases_df):
        from src.validation import validate_cases
        df = cleaned_cases_df.copy()
        df["Date"] = df["Date"].astype(str)
        with pytest.raises(ValueError, match="not datetime dtype"):
            validate_cases(df)


class TestValidateTesting:
    def test_valid_passes(self, cleaned_testing_df):
        from src.validation import validate_testing
        validate_testing(cleaned_testing_df)

    def test_empty_raises(self):
        from src.validation import validate_testing
        with pytest.raises(ValueError):
            validate_testing(pd.DataFrame())


class TestValidateVaccine:
    def test_valid_passes(self, cleaned_vaccine_df):
        from src.validation import validate_vaccine
        validate_vaccine(cleaned_vaccine_df)

    def test_empty_raises(self):
        from src.validation import validate_vaccine
        with pytest.raises(ValueError):
            validate_vaccine(pd.DataFrame())
