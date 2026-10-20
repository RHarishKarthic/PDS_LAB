"""
validation.py
-------------
Schema and content validation for all three datasets.
Raises descriptive ValueError exceptions on failure so the pipeline
fails loudly rather than silently producing wrong results.
"""
import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns after cleaning (post clean_*())
# ---------------------------------------------------------------------------
REQUIRED_CASES_COLS: List[str] = [
    "Date", "State", "Confirmed", "Cured", "Deaths",
    "DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries",
]
REQUIRED_TESTING_COLS: List[str] = [
    "Date", "State", "TotalSamples",
]
REQUIRED_VACCINE_COLS: List[str] = [
    "Date", "State", "Total Doses Administered",
]


def _check_cols(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{name}] Missing required columns after cleaning: {missing}"
        )


def _check_not_empty(df: pd.DataFrame, name: str) -> None:
    if len(df) == 0:
        raise ValueError(f"[{name}] DataFrame is empty after cleaning.")


def _check_date_col(df: pd.DataFrame, date_col: str, name: str) -> None:
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise ValueError(
            f"[{name}] Column '{date_col}' is not datetime dtype "
            f"(got {df[date_col].dtype})."
        )
    null_count = df[date_col].isnull().sum()
    if null_count > 0:
        raise ValueError(
            f"[{name}] Column '{date_col}' has {null_count} null values "
            "after cleaning."
        )


def _check_non_negative(
    df: pd.DataFrame, col: str, name: str, allow_zero: bool = True
) -> None:
    if col not in df.columns:
        return
    min_val = pd.to_numeric(df[col], errors="coerce").min()
    if not allow_zero and min_val <= 0:
        logger.warning(
            "[%s] Column '%s' has non-positive values (min=%s).", name, col, min_val
        )
    elif min_val < 0:
        raise ValueError(
            f"[{name}] Column '{col}' has negative values (min={min_val}) "
            "after cleaning. This indicates a cleaning error."
        )


def validate_cases(df: pd.DataFrame) -> None:
    """
    Validate cleaned cases DataFrame.

    Raises
    ------
    ValueError
        On any structural or content problem.
    """
    name = "Cases"
    _check_not_empty(df, name)
    _check_cols(df, REQUIRED_CASES_COLS, name)
    _check_date_col(df, "Date", name)
    for col in ["DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries"]:
        _check_non_negative(df, col, name)
    # Cumulative columns must be non-negative
    for col in ["Confirmed", "Cured", "Deaths"]:
        _check_non_negative(df, col, name)
    logger.info("[%s] Validation passed (%d rows).", name, len(df))


def validate_testing(df: pd.DataFrame) -> None:
    """Validate cleaned testing DataFrame."""
    name = "Testing"
    _check_not_empty(df, name)
    _check_cols(df, REQUIRED_TESTING_COLS, name)
    _check_date_col(df, "Date", name)
    _check_non_negative(df, "TotalSamples", name)
    logger.info("[%s] Validation passed (%d rows).", name, len(df))


def validate_vaccine(df: pd.DataFrame) -> None:
    """Validate cleaned vaccine DataFrame."""
    name = "Vaccine"
    _check_not_empty(df, name)
    _check_cols(df, REQUIRED_VACCINE_COLS, name)
    _check_date_col(df, "Date", name)
    _check_non_negative(df, "Total Doses Administered", name)
    logger.info("[%s] Validation passed (%d rows).", name, len(df))


def validate_all(
    cases: pd.DataFrame,
    testing: pd.DataFrame,
    vaccine: pd.DataFrame,
) -> None:
    """Run all three validators. Raises on first failure."""
    validate_cases(cases)
    validate_testing(testing)
    validate_vaccine(vaccine)
    logger.info("All dataset validations passed.")
