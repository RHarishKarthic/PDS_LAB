"""
analysis.py
-----------
Core analytical functions operating on cleaned pandas DataFrames.
All functions return DataFrames suitable for visualisation and reporting.

Key distinctions
----------------
- 'Confirmed', 'Cured', 'Deaths' are CUMULATIVE daily snapshots.
  Use 'DailyNewCases', 'DailyNewDeaths', 'DailyNewRecoveries' for flows.
- Rolling averages are computed on daily new counts, not cumulative totals.
- Anomaly detection uses a Z-score method on the 7-day rolling average
  of DailyNewCases, flagging points where Z > threshold.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State / region filtering helpers
# ---------------------------------------------------------------------------

def filter_state(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Return rows for a single state."""
    filtered = df[df["State"] == state].copy()
    if filtered.empty:
        logger.warning("No data found for state '%s'.", state)
    return filtered


def get_state_list(df: pd.DataFrame) -> list:
    """Return sorted unique state names."""
    return sorted(df["State"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Rolling averages
# ---------------------------------------------------------------------------

def compute_rolling_average(
    df: pd.DataFrame,
    value_col: str,
    window: int = 7,
    group_col: str = "State",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Compute a rolling (window-day) mean of *value_col* within each group.

    The result column is named '<value_col>_rolling<window>d'.
    Input must be sorted by group_col, date_col.
    """
    df = df.sort_values([group_col, date_col]).copy()
    new_col = f"{value_col}_rolling{window}d"
    df[new_col] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return df


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_peak_periods(
    df: pd.DataFrame,
    value_col: str = "DailyNewCases",
    group_col: str = "State",
    date_col: str = "Date",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Identify the top-N peak dates for each state by *value_col*.

    Returns
    -------
    pd.DataFrame
        Columns: State, Date, <value_col>
    """
    idx = df.groupby(group_col)[value_col].nlargest(top_n).reset_index(level=0)
    result = df.loc[idx.index, [group_col, date_col, value_col]].copy()
    return result.sort_values([group_col, value_col], ascending=[True, False])


# ---------------------------------------------------------------------------
# Weekly / Monthly aggregations
# ---------------------------------------------------------------------------

def weekly_summary(
    df: pd.DataFrame,
    value_cols: Optional[list] = None,
    group_col: str = "State",
) -> pd.DataFrame:
    """
    Aggregate daily new counts to weekly totals per state.

    Uses ISO year-week as the grouping key.
    """
    if value_cols is None:
        value_cols = ["DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries"]
    value_cols = [c for c in value_cols if c in df.columns]

    df = df.copy()
    df["YearWeek"] = df["Date"].dt.to_period("W").astype(str)
    df["WeekStart"] = df["Date"].dt.to_period("W").dt.start_time

    agg = {col: "sum" for col in value_cols}
    agg["Date"] = "min"
    result = (
        df.groupby([group_col, "YearWeek", "WeekStart"])
        .agg(agg)
        .reset_index()
        .rename(columns={"Date": "WeekDate"})
        .sort_values([group_col, "WeekStart"])
    )
    return result


def monthly_summary(
    df: pd.DataFrame,
    value_cols: Optional[list] = None,
    group_col: str = "State",
) -> pd.DataFrame:
    """Aggregate daily new counts to monthly totals per state."""
    if value_cols is None:
        value_cols = ["DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries"]
    value_cols = [c for c in value_cols if c in df.columns]

    df = df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    df["MonthStart"] = df["Date"].dt.to_period("M").dt.start_time

    agg = {col: "sum" for col in value_cols}
    result = (
        df.groupby([group_col, "YearMonth", "MonthStart"])
        .agg(agg)
        .reset_index()
        .sort_values([group_col, "MonthStart"])
    )
    return result


# ---------------------------------------------------------------------------
# State-level comparison table
# ---------------------------------------------------------------------------

def state_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table comparing all states across key metrics.

    Metrics:
    - TotalConfirmed    : max cumulative Confirmed per state
    - TotalDeaths       : max cumulative Deaths per state
    - TotalRecoveries   : max cumulative Cured per state
    - PeakDailyNewCases : single highest day of new cases
    - PeakDate          : date of peak daily new cases
    - DataStart         : earliest date in dataset for the state
    - DataEnd           : latest date in dataset for the state
    """
    grp = df.groupby("State")
    summary = pd.DataFrame({
        "TotalConfirmed": grp["Confirmed"].max(),
        "TotalDeaths": grp["Deaths"].max(),
        "TotalRecoveries": grp["Cured"].max(),
        "PeakDailyNewCases": grp["DailyNewCases"].max(),
        "DataStart": grp["Date"].min(),
        "DataEnd": grp["Date"].max(),
    }).reset_index()

    # PeakDate — date when DailyNewCases was highest.
    # idxmax() can return NaN if the entire group is NaN; guard against that.
    idx = grp["DailyNewCases"].idxmax()
    valid_idx = idx.dropna()  # keep as original index labels (avoid unsafe int cast)
    peak_dates = (
        df.loc[valid_idx, ["State", "Date"]]
        .rename(columns={"Date": "PeakDate"})
        .drop_duplicates("State")
    )
    summary = summary.merge(peak_dates, on="State", how="left")
    summary = summary.sort_values("TotalConfirmed", ascending=False).reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Percentage change
# ---------------------------------------------------------------------------

def pct_change_weekly(
    weekly_df: pd.DataFrame,
    value_col: str = "DailyNewCases",
    group_col: str = "State",
) -> pd.DataFrame:
    """Add a percentage-change column comparing each week to the previous."""
    weekly_df = weekly_df.sort_values([group_col, "WeekStart"]).copy()
    weekly_df[f"{value_col}_PctChange"] = (
        weekly_df.groupby(group_col)[value_col].pct_change() * 100
    )
    return weekly_df


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    value_col: str = "DailyNewCases",
    rolling_window: int = 7,
    z_threshold: float = 2.5,
    group_col: str = "State",
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Flag dates where the rolling-average Z-score of *value_col* exceeds
    *z_threshold* as anomalies.

    Method
    ------
    1. Compute a 7-day rolling mean (min_periods=3) within each state.
    2. Compute the rolling 7-day standard deviation.
    3. Z = (rolling_mean - overall_mean) / overall_std per state.
    4. Flag rows where |Z| > z_threshold as anomalies.

    This is a simple, reproducible, documented statistical method.
    It does not predict future anomalies — it identifies historical
    deviations in the computed rolling series.

    Returns
    -------
    pd.DataFrame
        Original columns plus 'RollingMean7d', 'ZScore', 'IsAnomaly'.
    """
    df = df.sort_values([group_col, date_col]).copy()
    rolling_col = f"{value_col}_rolling7d"

    df[rolling_col] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(7, min_periods=3).mean())
    )

    def _z_score(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sigma = series.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    df["ZScore"] = (
        df.groupby(group_col)[rolling_col].transform(_z_score)
    )
    df["IsAnomaly"] = df["ZScore"].abs() > z_threshold
    logger.info(
        "Anomaly detection: %d anomalous records (Z>%.1f) out of %d",
        df["IsAnomaly"].sum(),
        z_threshold,
        len(df),
    )
    return df


# ---------------------------------------------------------------------------
# Tamil Nadu focused analysis
# ---------------------------------------------------------------------------

def tn_detailed_analysis(
    cases: pd.DataFrame,
    testing: pd.DataFrame,
    vaccine: pd.DataFrame,
    rolling_window: int = 7,
    z_threshold: float = 2.5,
) -> dict:
    """
    Run the full analysis pipeline focused on Tamil Nadu.

    Returns a dict with keys:
        'cases_tn', 'testing_tn', 'vaccine_tn',
        'weekly_tn', 'monthly_tn',
        'anomalies_tn', 'state_comparison'
    """
    TN = "Tamil Nadu"

    cases_tn = filter_state(cases, TN)
    testing_tn = filter_state(testing, TN)
    vaccine_tn = filter_state(vaccine, TN)

    # Rolling averages
    cases_tn = compute_rolling_average(
        cases_tn, "DailyNewCases", rolling_window, group_col="State"
    )
    cases_tn = compute_rolling_average(
        cases_tn, "DailyNewDeaths", rolling_window, group_col="State"
    )

    # Summaries
    weekly_tn = weekly_summary(cases_tn)
    monthly_tn = monthly_summary(cases_tn)

    # Anomaly detection
    anomalies_tn = detect_anomalies(
        cases_tn, z_threshold=z_threshold
    )

    # State comparison table (all states)
    state_comp = state_comparison_table(cases)

    return {
        "cases_tn": cases_tn,
        "testing_tn": testing_tn,
        "vaccine_tn": vaccine_tn,
        "weekly_tn": weekly_tn,
        "monthly_tn": monthly_tn,
        "anomalies_tn": anomalies_tn,
        "state_comparison": state_comp,
    }
