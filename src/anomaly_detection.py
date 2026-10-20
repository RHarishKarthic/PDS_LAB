"""
anomaly_detection.py
--------------------
Standalone anomaly detection utilities used by both the dashboard
and the automated reporting pipeline.

Method: Z-Score on Rolling Mean
--------------------------------
1. Compute a rolling-N-day mean of DailyNewCases per state.
2. Compute the per-state mean and std of that rolling series.
3. Z = (rolling_value - state_mean) / state_std.
4. Flag dates where |Z| > threshold as anomalies.

This method is:
- Simple and reproducible on any dataset.
- Transparent: the threshold and formula are documented.
- Conservative: it identifies historical statistical deviations,
  NOT predictions of future outbreaks.
- Not a clinical diagnostic tool.  Real outbreak surveillance
  systems use additional epidemiological criteria.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def rolling_zscore_anomalies(
    df: pd.DataFrame,
    value_col: str = "DailyNewCases",
    rolling_window: int = 7,
    z_threshold: float = 2.5,
    group_col: str = "State",
    date_col: str = "Date",
    min_periods: int = 3,
) -> pd.DataFrame:
    """
    Detect anomalies using rolling Z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned cases (or testing) DataFrame.
    value_col : str
        Column to analyse.
    rolling_window : int
        Window size in days.
    z_threshold : float
        Absolute Z-score above which a point is flagged as anomalous.
    group_col : str
        Column used to group independent time series.
    date_col : str
        Date column.
    min_periods : int
        Minimum observations to compute rolling statistics.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added columns:
        'RollingMean', 'ZScore', 'IsAnomaly', 'AnomalyDirection'.
    """
    df = df.sort_values([group_col, date_col]).copy()
    rolling_col = f"Rolling{rolling_window}d_{value_col}"

    # Step 1: rolling mean per group
    df[rolling_col] = (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.rolling(rolling_window, min_periods=min_periods).mean())
    )

    # Step 2: Z-score — normalised within each group using group stats
    def _group_zscore(series: pd.Series) -> pd.Series:
        mu = series.mean()
        sigma = series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    df["ZScore"] = (
        df.groupby(group_col)[rolling_col]
        .transform(_group_zscore)
    )

    # Step 3: flag
    df["IsAnomaly"] = df["ZScore"].abs() > z_threshold
    df["AnomalyDirection"] = np.where(
        df["ZScore"] > z_threshold, "Spike",
        np.where(df["ZScore"] < -z_threshold, "Drop", "Normal")
    )

    n_anomalies = df["IsAnomaly"].sum()
    logger.info(
        "Anomaly detection on '%s': %d anomalies flagged (|Z|>%.2f, window=%d d)",
        value_col, n_anomalies, z_threshold, rolling_window,
    )
    return df


def anomaly_summary(anomaly_df: pd.DataFrame, group_col: str = "State") -> pd.DataFrame:
    """
    Build a summary of anomaly counts per state, including the peak anomaly date.

    Returns
    -------
    pd.DataFrame
    """
    anom = anomaly_df[anomaly_df["IsAnomaly"]].copy()
    if anom.empty:
        return pd.DataFrame(columns=[group_col, "AnomalyCount", "MaxZScore", "PeakAnomalyDate"])

    # Aggregate count and max Z-score per group
    summary = (
        anom.groupby(group_col)
        .agg(
            AnomalyCount=("IsAnomaly", "sum"),
            MaxZScore=("ZScore", "max"),
        )
        .reset_index()
    )

    # PeakAnomalyDate — date of the row with the maximum ZScore per group.
    # Done separately to avoid the closure-over-outer-frame bug in a lambda.
    peak_idx = anom.groupby(group_col)["ZScore"].idxmax()
    peak_dates = (
        anom.loc[peak_idx, [group_col, "Date"]]
        .rename(columns={"Date": "PeakAnomalyDate"})
        .reset_index(drop=True)
    )

    summary = summary.merge(peak_dates, on=group_col, how="left")
    summary = summary.sort_values("AnomalyCount", ascending=False).reset_index(drop=True)
    return summary
