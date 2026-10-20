"""
data_loader.py
--------------
Discovers and loads the three raw CSV datasets into pandas DataFrames.
Performs only light, non-destructive type coercion at load time.
Heavy cleaning is delegated to data_cleaning.py.
"""
import logging
import os
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def _read_csv_safe(path: str, encoding: str = "utf-8") -> pd.DataFrame:
    """Read a CSV, falling back to latin-1 encoding on decode errors."""
    try:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed for %s — retrying with latin-1", path)
        return pd.read_csv(path, encoding="latin-1", low_memory=False)


def load_cases(raw_dir: str, filename: str = "covid_19_india.csv") -> pd.DataFrame:
    """
    Load the state-wise daily COVID-19 cases dataset.

    Columns expected:
        Sno, Date, Time, State/UnionTerritory,
        ConfirmedIndianNational, ConfirmedForeignNational,
        Cured, Deaths, Confirmed

    Returns
    -------
    pd.DataFrame
        Raw cases DataFrame with 'Date' column parsed to datetime.
    """
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cases file not found: {path}\n"
            "Place covid_19_india.csv in the data/raw/ directory."
        )
    df = _read_csv_safe(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    _date_min = df["Date"].min().date() if df["Date"].notna().any() else "N/A"
    _date_max = df["Date"].max().date() if df["Date"].notna().any() else "N/A"
    logger.info(
        "Loaded cases: %d rows × %d cols  |  %s to %s",
        len(df),
        len(df.columns),
        _date_min,
        _date_max,
    )
    return df


def load_testing(
    raw_dir: str, filename: str = "StatewiseTestingDetails.csv"
) -> pd.DataFrame:
    """
    Load the state-wise COVID-19 testing dataset.

    Columns expected:
        Date, State, TotalSamples, Negative, Positive

    Returns
    -------
    pd.DataFrame
        Raw testing DataFrame with 'Date' column parsed to datetime.
    """
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Testing file not found: {path}\n"
            "Place StatewiseTestingDetails.csv in the data/raw/ directory."
        )
    df = _read_csv_safe(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    # Coerce numeric columns that may have been read as object
    for col in ["TotalSamples", "Negative", "Positive"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _date_min = df["Date"].min().date() if df["Date"].notna().any() else "N/A"
    _date_max = df["Date"].max().date() if df["Date"].notna().any() else "N/A"
    logger.info(
        "Loaded testing: %d rows × %d cols  |  %s to %s",
        len(df),
        len(df.columns),
        _date_min,
        _date_max,
    )
    return df


def load_vaccine(
    raw_dir: str, filename: str = "covid_vaccine_statewise.csv"
) -> pd.DataFrame:
    """
    Load the state-wise COVID-19 vaccination dataset.

    Key columns include:
        Updated On, State, Total Doses Administered,
        First Dose Administered, Second Dose Administered,
        Total Individuals Vaccinated

    Returns
    -------
    pd.DataFrame
        Raw vaccine DataFrame with 'Updated On' parsed to datetime.
    """
    path = os.path.join(raw_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vaccine file not found: {path}\n"
            "Place covid_vaccine_statewise.csv in the data/raw/ directory."
        )
    df = _read_csv_safe(path)
    df["Updated On"] = pd.to_datetime(
        df["Updated On"], format="%d/%m/%Y", errors="coerce"
    )
    _date_min = df["Updated On"].min().date() if df["Updated On"].notna().any() else "N/A"
    _date_max = df["Updated On"].max().date() if df["Updated On"].notna().any() else "N/A"
    logger.info(
        "Loaded vaccine: %d rows × %d cols  |  %s to %s",
        len(df),
        len(df.columns),
        _date_min,
        _date_max,
    )
    return df


def load_all(
    raw_dir: str,
    cases_file: str = "covid_19_india.csv",
    testing_file: str = "StatewiseTestingDetails.csv",
    vaccine_file: str = "covid_vaccine_statewise.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper: load all three datasets at once.

    Returns
    -------
    (cases_df, testing_df, vaccine_df)
    """
    cases = load_cases(raw_dir, cases_file)
    testing = load_testing(raw_dir, testing_file)
    vaccine = load_vaccine(raw_dir, vaccine_file)
    return cases, testing, vaccine
