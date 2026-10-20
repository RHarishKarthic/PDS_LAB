"""
tests/conftest.py
-----------------
Shared pytest fixtures using synthetic data only.
The actual project analysis uses real datasets — these fixtures
are for unit tests only.
"""

import pandas as pd
import pytest
import numpy as np


@pytest.fixture
def raw_cases_df():
    """Minimal synthetic raw cases DataFrame mimicking covid_19_india.csv schema."""
    return pd.DataFrame({
        "Sno": [1, 2, 3, 4, 5, 6],
        "Date": ["2020-03-01", "2020-03-02", "2020-03-03",
                 "2020-03-01", "2020-03-02", "2020-03-03"],
        "Time": ["6:00 PM"] * 6,
        "State/UnionTerritory": ["Tamil Nadu", "Tamil Nadu", "Tamil Nadu",
                                  "Kerala", "Kerala", "Kerala"],
        "ConfirmedIndianNational": ["1", "2", "3", "0", "1", "2"],
        "ConfirmedForeignNational": ["0"] * 6,
        "Cured": [0, 0, 1, 0, 1, 2],
        "Deaths": [0, 0, 0, 0, 0, 1],
        "Confirmed": [1, 2, 3, 0, 1, 2],
    })


@pytest.fixture
def raw_testing_df():
    """Minimal synthetic raw testing DataFrame."""
    return pd.DataFrame({
        "Date": ["2020-04-01", "2020-04-02", "2020-04-03",
                 "2020-04-01", "2020-04-02", "2020-04-03"],
        "State": ["Tamil Nadu"] * 3 + ["Kerala"] * 3,
        "TotalSamples": [100.0, 200.0, 350.0, 50.0, 120.0, 200.0],
        "Negative": ["90", "180", "310", "45", "110", "180"],
        "Positive": [10.0, 20.0, 40.0, 5.0, 10.0, 20.0],
    })


@pytest.fixture
def raw_vaccine_df():
    """Minimal synthetic raw vaccine DataFrame."""
    return pd.DataFrame({
        "Updated On": ["16/01/2021", "17/01/2021", "18/01/2021",
                       "16/01/2021", "17/01/2021", "18/01/2021"],
        "State": ["Tamil Nadu"] * 3 + ["Kerala"] * 3,
        "Total Doses Administered": [1000.0, 2500.0, 4000.0, 500.0, 1200.0, 2000.0],
        "Sessions": [10.0, 25.0, 40.0, 5.0, 12.0, 20.0],
        " Sites ": [8.0, 20.0, 30.0, 4.0, 10.0, 16.0],
        "First Dose Administered": [1000.0, 2500.0, 4000.0, 500.0, 1200.0, 2000.0],
        "Second Dose Administered": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Male (Doses Administered)": [500.0, 1200.0, 2000.0, 250.0, 600.0, 1000.0],
        "Female (Doses Administered)": [500.0, 1300.0, 2000.0, 250.0, 600.0, 1000.0],
        "Transgender (Doses Administered)": [0.0] * 6,
        " Covaxin (Doses Administered)": [100.0, 250.0, 400.0, 50.0, 120.0, 200.0],
        "CoviShield (Doses Administered)": [900.0, 2250.0, 3600.0, 450.0, 1080.0, 1800.0],
        "Sputnik V (Doses Administered)": [np.nan] * 6,
        "AEFI": [0.0] * 6,
        "18-44 Years (Doses Administered)": [np.nan] * 6,
        "45-60 Years (Doses Administered)": [np.nan] * 6,
        "60+ Years (Doses Administered)": [np.nan] * 6,
        "18-44 Years(Individuals Vaccinated)": [np.nan] * 6,
        "45-60 Years(Individuals Vaccinated)": [np.nan] * 6,
        "60+ Years(Individuals Vaccinated)": [np.nan] * 6,
        "Male(Individuals Vaccinated)": [np.nan] * 6,
        "Female(Individuals Vaccinated)": [np.nan] * 6,
        "Transgender(Individuals Vaccinated)": [np.nan] * 6,
        "Total Individuals Vaccinated": [1000.0, 2500.0, 4000.0, 500.0, 1200.0, 2000.0],
    })


@pytest.fixture
def cleaned_cases_df(raw_cases_df):
    """Return cleaned version of synthetic cases data."""
    from src.data_cleaning import clean_cases
    df = raw_cases_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return clean_cases(df)


@pytest.fixture
def cleaned_testing_df(raw_testing_df):
    from src.data_cleaning import clean_testing
    df = raw_testing_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["TotalSamples", "Negative", "Positive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return clean_testing(df)


@pytest.fixture
def cleaned_vaccine_df(raw_vaccine_df):
    from src.data_cleaning import clean_vaccine
    df = raw_vaccine_df.copy()
    df["Updated On"] = pd.to_datetime(df["Updated On"], format="%d/%m/%Y")
    return clean_vaccine(df)
