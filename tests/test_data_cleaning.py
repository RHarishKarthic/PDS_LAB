"""
tests/test_data_cleaning.py
---------------------------
Unit tests for data_cleaning.py using synthetic fixtures.
"""

import pandas as pd
import pytest


class TestCleanCases:
    def test_output_columns(self, cleaned_cases_df):
        for col in ["Date", "State", "Confirmed", "Cured", "Deaths",
                    "DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries"]:
            assert col in cleaned_cases_df.columns

    def test_no_null_date(self, cleaned_cases_df):
        assert cleaned_cases_df["Date"].isnull().sum() == 0

    def test_no_null_state(self, cleaned_cases_df):
        assert cleaned_cases_df["State"].isnull().sum() == 0

    def test_daily_new_cases_non_negative(self, cleaned_cases_df):
        assert (cleaned_cases_df["DailyNewCases"] >= 0).all()

    def test_daily_new_deaths_non_negative(self, cleaned_cases_df):
        assert (cleaned_cases_df["DailyNewDeaths"] >= 0).all()

    def test_daily_new_recoveries_non_negative(self, cleaned_cases_df):
        assert (cleaned_cases_df["DailyNewRecoveries"] >= 0).all()

    def test_first_diff_correct(self, cleaned_cases_df):
        """First difference of cumulative confirmed = daily new cases (where positive)."""
        tn = cleaned_cases_df[cleaned_cases_df["State"] == "Tamil Nadu"].sort_values("Date")
        # Confirmed values: 1, 2, 3 → daily diffs: NaN (→0), 1, 1
        assert tn["DailyNewCases"].iloc[1] == 1
        assert tn["DailyNewCases"].iloc[2] == 1

    def test_no_old_state_name_column(self, cleaned_cases_df):
        assert "State/UnionTerritory" not in cleaned_cases_df.columns

    def test_state_alias_resolution(self):
        """Bihar**** should become Bihar."""
        import pandas as pd
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Sno": [1], "Date": [pd.Timestamp("2020-03-01")], "Time": ["6:00 PM"],
            "State/UnionTerritory": ["Bihar****"],
            "ConfirmedIndianNational": ["1"], "ConfirmedForeignNational": ["0"],
            "Cured": [0], "Deaths": [0], "Confirmed": [1],
        })
        cleaned = clean_cases(df)
        assert "Bihar" in cleaned["State"].values

    def test_unassignable_state_dropped(self):
        """'Cases being reassigned to states' should be dropped."""
        import pandas as pd
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Sno": [1], "Date": [pd.Timestamp("2020-03-01")], "Time": ["6:00 PM"],
            "State/UnionTerritory": ["Cases being reassigned to states"],
            "ConfirmedIndianNational": ["1"], "ConfirmedForeignNational": ["0"],
            "Cured": [0], "Deaths": [0], "Confirmed": [1],
        })
        cleaned = clean_cases(df)
        assert len(cleaned) == 0

    def test_duplicate_state_date_removed(self):
        """Duplicate State+Date rows should be dropped."""
        import pandas as pd
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Sno": [1, 2], "Date": [pd.Timestamp("2020-03-01")] * 2,
            "Time": ["6:00 PM"] * 2,
            "State/UnionTerritory": ["Tamil Nadu"] * 2,
            "ConfirmedIndianNational": ["1"] * 2, "ConfirmedForeignNational": ["0"] * 2,
            "Cured": [0, 0], "Deaths": [0, 0], "Confirmed": [1, 1],
        })
        cleaned = clean_cases(df)
        assert len(cleaned) == 1

    def test_temporal_columns_added(self, cleaned_cases_df):
        for col in ["Year", "Month", "Week", "YearMonth", "YearWeek"]:
            assert col in cleaned_cases_df.columns

    def test_negative_diff_capped(self):
        """A decrease in cumulative Confirmed should produce DailyNewCases=0, not negative."""
        import pandas as pd
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Sno": [1, 2], "Date": [pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-02")],
            "Time": ["6:00 PM"] * 2,
            "State/UnionTerritory": ["Tamil Nadu"] * 2,
            "ConfirmedIndianNational": ["5", "3"], "ConfirmedForeignNational": ["0"] * 2,
            "Cured": [0, 0], "Deaths": [0, 0], "Confirmed": [5, 3],  # drop from 5→3
        })
        cleaned = clean_cases(df)
        assert (cleaned["DailyNewCases"] >= 0).all()
        # Flag should be set
        assert "DailyNewCases_was_negative" in cleaned.columns
        assert cleaned["DailyNewCases_was_negative"].any()


class TestCleanTesting:
    def test_output_columns(self, cleaned_testing_df):
        for col in ["Date", "State", "TotalSamples", "DailyNewSamples"]:
            assert col in cleaned_testing_df.columns

    def test_positivity_rate_in_0_100(self, cleaned_testing_df):
        pr = cleaned_testing_df["PositivityRate"].dropna()
        assert (pr >= 0).all()
        assert (pr <= 100).all()

    def test_no_null_date(self, cleaned_testing_df):
        assert cleaned_testing_df["Date"].isnull().sum() == 0


class TestCleanVaccine:
    def test_output_columns(self, cleaned_vaccine_df):
        for col in ["Date", "State", "Total Doses Administered", "DailyNewDoses"]:
            assert col in cleaned_vaccine_df.columns

    def test_india_aggregate_removed(self, cleaned_vaccine_df):
        assert "India" not in cleaned_vaccine_df["State"].values

    def test_date_is_datetime(self, cleaned_vaccine_df):
        assert pd.api.types.is_datetime64_any_dtype(cleaned_vaccine_df["Date"])

    def test_daily_new_doses_non_negative(self, cleaned_vaccine_df):
        assert (cleaned_vaccine_df["DailyNewDoses"] >= 0).all()


class TestDataQualityReport:
    def test_report_shape(self, cleaned_cases_df, cleaned_testing_df, cleaned_vaccine_df):
        from src.data_cleaning import data_quality_report
        report = data_quality_report(cleaned_cases_df, cleaned_testing_df, cleaned_vaccine_df)
        assert len(report) == 3
        for col in ["Dataset", "Rows", "Columns", "Date Range Start", "Date Range End"]:
            assert col in report.columns


class TestDataCleaningRobustness:
    def test_comma_separated_numbers_parsed(self):
        """Numbers containing commas should be correctly parsed as numeric instead of turning to NaN/0."""
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Date": ["2020-03-01"],
            "State/UnionTerritory": ["Goa"],
            "Confirmed": ["1,500"],
            "Cured": ["1,200"],
            "Deaths": ["50"],
        })
        cleaned = clean_cases(df)
        assert cleaned["Confirmed"].iloc[0] == 1500
        assert cleaned["Cured"].iloc[0] == 1200
        assert cleaned["Deaths"].iloc[0] == 50

    def test_unparsed_string_date_handled(self):
        """clean_cases should safely parse date strings without requiring pre-parsed datetime input."""
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Date": ["2020-05-10"],
            "State/UnionTerritory": ["Kerala"],
            "Confirmed": [10], "Cured": [5], "Deaths": [1],
        })
        cleaned = clean_cases(df)
        assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
        assert cleaned["Year"].iloc[0] == 2020
        assert cleaned["Month"].iloc[0] == 5

    def test_blank_and_whitespace_state_dropped(self):
        """Empty or whitespace-only state strings should be treated as unassignable and dropped."""
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Date": ["2020-05-10", "2020-05-11"],
            "State/UnionTerritory": ["   ", ""],
            "Confirmed": [10, 20], "Cured": [5, 10], "Deaths": [1, 2],
        })
        cleaned = clean_cases(df)
        assert len(cleaned) == 0

    def test_arbitrary_asterisk_state_cleaned(self):
        """State names with trailing asterisks should be automatically stripped."""
        from src.data_cleaning import clean_cases
        df = pd.DataFrame({
            "Date": ["2020-05-10"],
            "State/UnionTerritory": ["Odisha***"],
            "Confirmed": [10], "Cured": [5], "Deaths": [1],
        })
        cleaned = clean_cases(df)
        assert cleaned["State"].iloc[0] == "Odisha"

    def test_vaccine_string_numeric_coercion(self):
        """clean_vaccine should safely handle string numeric columns with commas."""
        from src.data_cleaning import clean_vaccine
        df = pd.DataFrame({
            "Updated On": ["16/01/2021", "17/01/2021"],
            "State": ["Goa", "Goa"],
            "Total Doses Administered": ["1,000", "2,500"],
        })
        cleaned = clean_vaccine(df)
        assert cleaned["Total Doses Administered"].iloc[0] == 1000.0
        assert cleaned["Total Doses Administered"].iloc[1] == 2500.0
        assert cleaned["DailyNewDoses"].iloc[1] == 1500.0

    def test_temporal_columns_parity(self, cleaned_testing_df, cleaned_vaccine_df):
        """Ensure Week and YearWeek are present in testing and vaccine datasets as well."""
        for col in ["Year", "Month", "Week", "YearMonth", "YearWeek"]:
            assert col in cleaned_testing_df.columns
            assert col in cleaned_vaccine_df.columns

