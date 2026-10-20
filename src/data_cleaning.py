"""
data_cleaning.py
----------------
Cleans and standardises the three raw DataFrames.
All cleaning decisions are documented in-line.
The original raw files are never modified.

Cleaning rules applied
-----------------------
Cases (covid_19_india.csv):
  * Drop rows with null Date or null State/UnionTerritory.
  * Rename 'State/UnionTerritory' → 'State' for consistency.
  * Strip whitespace from State names; strip trailing asterisks; apply known alias corrections.
  * Drop obviously corrupt state tokens (e.g. 'Bihar****',
    'Cases being reassigned to states').
  * Drop the Sno, Time, ConfirmedIndianNational, ConfirmedForeignNational
    columns (not needed for analysis).
  * Sort by State, Date; drop exact duplicates.
  * The 'Confirmed', 'Cured', 'Deaths' columns are CUMULATIVE daily
    snapshots — they must NOT be summed across dates for a total.
  * Compute DailyNewCases, DailyNewDeaths, DailyNewRecoveries as
    first-differences within each State, capped at 0 to remove
    negative corrections (artefacts of state data revisions).
  * Flag negative differences before capping so they are auditable.
  * Generate standardized temporal features: Year, Month, Week, YearMonth, YearWeek.

Testing (StatewiseTestingDetails.csv):
  * Drop rows with null Date or State.
  * Strip whitespace from State names and apply state normalization.
  * Drop duplicate State+Date rows.
  * Negative and Positive are cumulative where present; treat with the
    same first-difference logic.
  * Compute DailyNewSamples as first-difference of cumulative TotalSamples.
  * Compute PositivityRate (clipped between 0% and 100%).
  * Generate standardized temporal features: Year, Month, Week, YearMonth, YearWeek.

Vaccine (covid_vaccine_statewise.csv):
  * Drop 'India' aggregate rows — state-level analysis only.
  * Rename 'Updated On' → 'Date'.
  * Strip whitespace from column names and State names.
  * Coerce numeric columns safely (handling any comma separators).
  * Compute DailyNewDoses as first-difference of cumulative Total Doses Administered.
  * The 'Total Doses Administered' column is CUMULATIVE — document this.
  * Generate standardized temporal features: Year, Month, Week, YearMonth, YearWeek.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known state-name aliases encountered in the datasets
# ---------------------------------------------------------------------------
STATE_ALIASES: Dict[str, Optional[str]] = {
    "Bihar****": "Bihar",
    "Telangana ***": "Telangana",
    "Telangana***": "Telangana",
    "Telengana": "Telangana",
    "Maharashtra***": "Maharashtra",
    "Madhya Pradesh***": "Madhya Pradesh",
    "Karanataka": "Karnataka",
    "Himanchal Pradesh": "Himachal Pradesh",
    "Cases being reassigned to states": None,   # will be dropped
    "Unassigned": None,
    "Miscellaneous": None,
    "State Unassigned": None,
    "Dadra and Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman and Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Daman & Diu": "Dadra and Nagar Haveli and Daman and Diu",
    "Dadar Nagar Haveli": "Dadra and Nagar Haveli and Daman and Diu",
}


def _standardise_state(series: pd.Series) -> pd.Series:
    """Strip whitespace, trailing asterisks, apply alias map, return normalised Series."""
    mask_na = series.isna()
    s = series.fillna("").astype(str).str.strip().str.rstrip("*").str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    s[mask_na] = np.nan
    s = s.map(lambda x: STATE_ALIASES.get(x, x) if pd.notna(x) else np.nan)
    return s


def _parse_date_safe(series: pd.Series, preferred_format: Optional[str] = None) -> pd.Series:
    """Safely parse series to datetime64[ns], preserving already-parsed datetime series."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if preferred_format:
        parsed = pd.to_datetime(series, format=preferred_format, errors="coerce")
        if parsed.notna().sum() == len(series.dropna()):
            return parsed
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series, fill_value=None, to_int: bool = False) -> pd.Series:
    """Safely strip commas and whitespace and convert series to numeric."""
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False).str.strip()
    s = pd.to_numeric(series, errors="coerce")
    if fill_value is not None:
        s = s.fillna(fill_value)
    if to_int:
        s = s.astype(int)
    return s


def _add_temporal_columns(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Add unified temporal columns (Year, Month, Week, YearMonth, YearWeek)."""
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Week"] = df[date_col].dt.isocalendar().week.astype(int)
    df["YearMonth"] = df[date_col].dt.to_period("M").astype(str)
    df["YearWeek"] = df[date_col].dt.to_period("W").astype(str)
    return df


def _first_diff_non_negative(
    df: pd.DataFrame, group_col: str, date_col: str, value_col: str, new_col: str
) -> pd.DataFrame:
    """
    Within each group, compute the first difference of *value_col*
    (sorted by date), store in *new_col*, and floor negative differences
    at 0 (they represent state data-revision artefacts).

    The first row of each group will have NaN from .diff(); this is
    filled with 0 because there is no prior day to measure against.

    A boolean '<new_col>_was_negative' flag is added for auditability.
    """
    df = df.sort_values([group_col, date_col])
    diff = df.groupby(group_col)[value_col].diff()
    # Flag before capping (NaN rows are neither positive nor negative)
    df[new_col + "_was_negative"] = diff.fillna(0) < 0
    # Clip negatives to 0, then fill first-row NaN with 0
    df[new_col] = diff.clip(lower=0).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def clean_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the covid_19_india.csv DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_cases().

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with daily new-case columns added.
    """
    logger.info("Cleaning cases: start shape %s", df.shape)
    df = df.copy()

    # Rename for consistency
    df = df.rename(columns={"State/UnionTerritory": "State"})

    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = _parse_date_safe(df["Date"], preferred_format="%Y-%m-%d")

    # Drop rows missing critical fields
    before = len(df)
    df = df.dropna(subset=["Date", "State"])
    logger.info("Dropped %d rows with null Date or State", before - len(df))

    # Standardise state names
    df["State"] = _standardise_state(df["State"])

    # Drop rows whose State was mapped to None (unassignable)
    before = len(df)
    df = df[df["State"].notna()]
    logger.info("Dropped %d unassignable-state rows", before - len(df))

    # Drop unnecessary columns
    drop_cols = [c for c in ["Sno", "Time", "ConfirmedIndianNational",
                              "ConfirmedForeignNational"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Ensure numeric types
    for col in ["Confirmed", "Cured", "Deaths"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col], fill_value=0, to_int=True)

    # Sort and deduplicate (keep last record per State+Date)
    df = df.sort_values(["State", "Date"])
    before = len(df)
    df = df.drop_duplicates(subset=["State", "Date"], keep="last")
    logger.info("Removed %d duplicate State+Date rows", before - len(df))

    # Compute daily new counts from cumulative columns
    for cum_col, new_col in [
        ("Confirmed", "DailyNewCases"),
        ("Deaths", "DailyNewDeaths"),
        ("Cured", "DailyNewRecoveries"),
    ]:
        df = _first_diff_non_negative(df, "State", "Date", cum_col, new_col)

    # Add temporal columns
    df = _add_temporal_columns(df, "Date")

    df = df.reset_index(drop=True)
    logger.info("Cleaned cases: final shape %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
def clean_testing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the StatewiseTestingDetails.csv DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_testing().

    Returns
    -------
    pd.DataFrame
        Cleaned testing DataFrame.
    """
    logger.info("Cleaning testing: start shape %s", df.shape)
    df = df.copy()

    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = _parse_date_safe(df["Date"], preferred_format="%Y-%m-%d")

    before = len(df)
    df = df.dropna(subset=["Date", "State"])
    logger.info("Dropped %d rows with null Date or State", before - len(df))

    df["State"] = _standardise_state(df["State"])
    before = len(df)
    df = df[df["State"].notna()]
    logger.info("Dropped %d unassignable-state rows", before - len(df))

    # Ensure numeric
    for col in ["TotalSamples", "Negative", "Positive"]:
        if col in df.columns:
            df[col] = _coerce_numeric(df[col])

    # Remove duplicate State+Date rows
    df = df.sort_values(["State", "Date"])
    before = len(df)
    df = df.drop_duplicates(subset=["State", "Date"], keep="last")
    logger.info("Removed %d duplicate State+Date rows", before - len(df))

    # Daily new samples tested (TotalSamples is cumulative)
    df = _first_diff_non_negative(df, "State", "Date", "TotalSamples", "DailyNewSamples")

    # Positivity rate — only where both values are available and TotalSamples > 0
    mask = df["TotalSamples"].notna() & (df["TotalSamples"] > 0) & df["Positive"].notna()
    df["PositivityRate"] = np.nan
    df.loc[mask, "PositivityRate"] = (
        (df.loc[mask, "Positive"] / df.loc[mask, "TotalSamples"] * 100).clip(lower=0.0, upper=100.0)
    )

    # Add temporal columns
    df = _add_temporal_columns(df, "Date")

    df = df.reset_index(drop=True)
    logger.info("Cleaned testing: final shape %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Vaccine
# ---------------------------------------------------------------------------
def clean_vaccine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the covid_vaccine_statewise.csv DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of data_loader.load_vaccine().

    Returns
    -------
    pd.DataFrame
        Cleaned vaccine DataFrame.

    Notes
    -----
    'Total Doses Administered' is CUMULATIVE — the last value per state
    in a given period represents the cumulative count, not a daily count.
    """
    logger.info("Cleaning vaccine: start shape %s", df.shape)
    df = df.copy()

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Rename date column
    df = df.rename(columns={"Updated On": "Date"})

    # Ensure Date is datetime
    if "Date" in df.columns:
        df["Date"] = _parse_date_safe(df["Date"], preferred_format="%d/%m/%Y")

    before = len(df)
    df = df.dropna(subset=["Date", "State"])
    logger.info("Dropped %d rows with null Date/State", before - len(df))

    # Drop India aggregate row — it mixes all states
    before = len(df)
    df = df[df["State"] != "India"]
    logger.info("Dropped %d 'India' aggregate rows", before - len(df))

    df["State"] = _standardise_state(df["State"])
    before = len(df)
    df = df[df["State"].notna()]
    logger.info("Dropped %d unassignable-state rows", before - len(df))

    # Coerce numeric columns
    numeric_cols = [c for c in df.columns if c not in ["Date", "State"]]
    for col in numeric_cols:
        df[col] = _coerce_numeric(df[col])

    df = df.sort_values(["State", "Date"])
    before = len(df)
    df = df.drop_duplicates(subset=["State", "Date"], keep="last")
    logger.info("Removed %d duplicate State+Date rows", before - len(df))

    # Compute daily new doses from cumulative column
    df = _first_diff_non_negative(
        df, "State", "Date", "Total Doses Administered", "DailyNewDoses"
    )

    # Add temporal columns
    df = _add_temporal_columns(df, "Date")

    df = df.reset_index(drop=True)
    logger.info("Cleaned vaccine: final shape %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def clean_all(
    cases_raw: pd.DataFrame,
    testing_raw: pd.DataFrame,
    vaccine_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean all three DataFrames and return them as a tuple."""
    return (
        clean_cases(cases_raw),
        clean_testing(testing_raw),
        clean_vaccine(vaccine_raw),
    )


# ---------------------------------------------------------------------------
# Data quality report
# ---------------------------------------------------------------------------
def data_quality_report(
    cases: pd.DataFrame,
    testing: pd.DataFrame,
    vaccine: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a concise data-quality summary table.

    Returns
    -------
    pd.DataFrame
        One row per dataset with quality metrics.
    """
    rows = []
    for name, df in [("Cases", cases), ("Testing", testing), ("Vaccine", vaccine)]:
        date_col = "Date"
        has_dates = date_col in df.columns and len(df) > 0 and df[date_col].notna().any()
        null_pct = df.isnull().mean().mean() * 100 if len(df) > 0 else 0.0
        rows.append({
            "Dataset": name,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Date Range Start": df[date_col].min().date() if has_dates else "N/A",
            "Date Range End": df[date_col].max().date() if has_dates else "N/A",
            "Missing Values (%)": round(null_pct, 2) if not np.isnan(null_pct) else 0.0,
            "Duplicate Rows Removed": "Yes (per State+Date)",
        })
    return pd.DataFrame(rows)
