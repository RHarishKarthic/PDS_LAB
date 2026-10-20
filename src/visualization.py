"""
visualization.py
----------------
Generates and saves all project charts using matplotlib / seaborn.
Every chart is saved as a PNG in outputs/charts/.

Each function:
  - Takes pandas DataFrames as input (never raw files).
  - Returns the figure path for logging / dashboard use.
  - Has proper axis labels, titles, legends, and units.
  - Is deterministic (same data → same chart).
"""

import logging
import os
from typing import Optional, List

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server / headless use
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style="darkgrid", palette="muted")
PALETTE = {
    "cases": "#E63946",
    "deaths": "#6D6875",
    "recoveries": "#2A9D8F",
    "doses": "#E9C46A",
    "sequential": "#264653",
    "threaded": "#2A9D8F",
    "spark": "#E9C46A",
    "anomaly": "#E63946",
    "gpu": "#9B5DE5",
}


def _save(fig: plt.Figure, path: str) -> str:
    """Save figure to path and close it. Returns path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# 1. Daily new cases over time (single state)
# ---------------------------------------------------------------------------

def plot_daily_cases(
    df: pd.DataFrame,
    state: str,
    rolling_col: Optional[str] = None,
    out_dir: str = "outputs/charts",
) -> str:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(df["Date"], df["DailyNewCases"], color=PALETTE["cases"],
           alpha=0.4, width=1, label="Daily New Cases")
    if rolling_col and rolling_col in df.columns:
        ax.plot(df["Date"], df[rolling_col], color="#B5182B",
                linewidth=2.2, label=f"7-Day Rolling Average")
    ax.set_title(f"Daily New COVID-19 Cases — {state}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("New Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    ax.legend()
    fname = f"daily_cases_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 2. Daily deaths over time
# ---------------------------------------------------------------------------

def plot_daily_deaths(
    df: pd.DataFrame,
    state: str,
    rolling_col: Optional[str] = None,
    out_dir: str = "outputs/charts",
) -> str:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(df["Date"], df["DailyNewDeaths"], color=PALETTE["deaths"],
           alpha=0.5, width=1, label="Daily New Deaths")
    if rolling_col and rolling_col in df.columns:
        ax.plot(df["Date"], df[rolling_col], color="#3D1A41",
                linewidth=2.2, label="7-Day Rolling Average")
    ax.set_title(f"Daily COVID-19 Deaths — {state}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Deaths")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    ax.legend()
    fname = f"daily_deaths_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 3. Monthly trend bar chart
# ---------------------------------------------------------------------------

def plot_monthly_trend(
    monthly_df: pd.DataFrame,
    state: str,
    value_col: str = "DailyNewCases",
    out_dir: str = "outputs/charts",
) -> str:
    df = monthly_df[monthly_df["State"] == state].copy()
    df = df.sort_values("MonthStart")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(df["YearMonth"], df[value_col], color=PALETTE["cases"], alpha=0.8)
    ax.set_title(f"Monthly New Cases — {state}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total New Cases")
    ax.tick_params(axis="x", rotation=45)
    fname = f"monthly_trend_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 4. State comparison — top N bar chart
# ---------------------------------------------------------------------------

def plot_state_comparison(
    state_comp: pd.DataFrame,
    top_n: int = 15,
    out_dir: str = "outputs/charts",
) -> str:
    df = state_comp.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df["State"][::-1], df["TotalConfirmed"][::-1],
                   color=PALETTE["cases"], alpha=0.85)
    ax.set_title(f"Top {top_n} States by Total Confirmed COVID-19 Cases", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Confirmed Cases (Cumulative Peak)")
    ax.set_ylabel("State")
    # Annotate bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{int(w):,}", va="center", fontsize=8)
    fname = f"state_comparison_top{top_n}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 5. Anomaly detection chart
# ---------------------------------------------------------------------------

def plot_anomalies(
    anomaly_df: pd.DataFrame,
    state: str,
    value_col: str = "DailyNewCases",
    out_dir: str = "outputs/charts",
) -> str:
    df = anomaly_df[anomaly_df["State"] == state].copy().sort_values("Date")
    rolling_col = [c for c in df.columns if "rolling" in c.lower() and "7" in c]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    # Top: raw values + rolling average
    ax = axes[0]
    ax.plot(df["Date"], df[value_col], alpha=0.4, color=PALETTE["cases"], linewidth=1, label=value_col)
    if rolling_col:
        ax.plot(df["Date"], df[rolling_col[0]], color="#B5182B", linewidth=2, label="7-Day Rolling Avg")
    # Mark anomalies
    anom = df[df["IsAnomaly"]]
    ax.scatter(anom["Date"], anom[value_col], color=PALETTE["anomaly"],
               zorder=5, s=60, label="Anomaly", marker="^")
    ax.set_ylabel(value_col)
    ax.set_title(f"Anomaly Detection — {state} — Z-Score Method", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    # Bottom: Z-score series
    ax2 = axes[1]
    ax2.plot(df["Date"], df["ZScore"], color="#264653", linewidth=1.5, label="Z-Score")
    ax2.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    # Determine threshold value
    threshold_val = df[df["IsAnomaly"]]["ZScore"].abs().min() if not anom.empty else 2.5
    ax2.axhline(threshold_val, color=PALETTE["anomaly"], linewidth=1.5, linestyle="--", label=f"Threshold ±{threshold_val:.1f}")
    ax2.axhline(-threshold_val, color=PALETTE["anomaly"], linewidth=1.5, linestyle="--")
    ax2.fill_between(df["Date"], df["ZScore"], 0, where=df["IsAnomaly"],
                     color=PALETTE["anomaly"], alpha=0.3)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Z-Score")
    ax2.legend(fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()

    fname = f"anomalies_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 6. Vaccination trend
# ---------------------------------------------------------------------------

def plot_vaccination(
    vaccine_df: pd.DataFrame,
    state: str,
    out_dir: str = "outputs/charts",
) -> str:
    df = vaccine_df[vaccine_df["State"] == state].copy().sort_values("Date")
    if df.empty:
        logger.warning("No vaccine data for state '%s'.", state)
        return ""
    fig, ax = plt.subplots(figsize=(13, 5))
    if "DailyNewDoses" in df.columns:
        ax.bar(df["Date"], df["DailyNewDoses"], color=PALETTE["doses"],
               alpha=0.7, width=1, label="Daily New Doses")
    if "Total Doses Administered" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["Date"], df["Total Doses Administered"], color="#E76F51",
                 linewidth=2, label="Cumulative Doses")
        ax2.set_ylabel("Cumulative Doses Administered")
        ax2.legend(loc="upper left")
    ax.set_title(f"COVID-19 Vaccination — {state}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily New Doses")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    fig.autofmt_xdate()
    ax.legend(loc="upper right")
    fname = f"vaccination_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 7. Benchmark comparison charts
# ---------------------------------------------------------------------------

def plot_benchmark(
    benchmark_df: pd.DataFrame,
    out_dir: str = "outputs/charts",
) -> List[str]:
    """Generate execution-time and speedup comparison charts."""
    paths = []

    # a) Execution time bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    df = benchmark_df.dropna(subset=["MedianElapsedSeconds"])
    colors = [PALETTE.get(k.split()[0].lower(), "#888888") for k in df["Implementation"]]
    bars = ax.bar(df["Implementation"], df["MedianElapsedSeconds"], color=colors, alpha=0.85)
    ax.set_title("Median Execution Time Comparison\n(3 runs, local machine)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Median Elapsed Time (seconds)")
    ax.set_xlabel("Implementation")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.2f}s", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    p = _save(fig, os.path.join(out_dir, "benchmark_execution_time.png"))
    paths.append(p)

    # b) Speedup bar chart
    if "SpeedupVsSequential" in df.columns:
        speedup_df = df.dropna(subset=["SpeedupVsSequential"])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(speedup_df["Implementation"], speedup_df["SpeedupVsSequential"],
               color=[PALETTE.get(k.split()[0].lower(), "#888") for k in speedup_df["Implementation"]],
               alpha=0.85)
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label="Speedup = 1 (baseline)")
        ax.set_title("Speedup vs Sequential Baseline", fontsize=12, fontweight="bold")
        ax.set_ylabel("Speedup Factor (×)")
        ax.set_xlabel("Implementation")
        ax.legend()
        ax.tick_params(axis="x", rotation=15)
        p = _save(fig, os.path.join(out_dir, "benchmark_speedup.png"))
        paths.append(p)

    return paths


# ---------------------------------------------------------------------------
# 8. Testing heatmap (positivity rates by month)
# ---------------------------------------------------------------------------

def plot_testing_heatmap(
    testing_df: pd.DataFrame,
    state: str,
    out_dir: str = "outputs/charts",
) -> str:
    df = testing_df[testing_df["State"] == state].copy()
    if df.empty or "PositivityRate" not in df.columns:
        return ""
    df["YearMonth"] = df["Date"].dt.to_period("M").astype(str)
    pivot = df.groupby("YearMonth")["PositivityRate"].mean().reset_index()
    pivot = pivot.set_index("YearMonth").T

    fig, ax = plt.subplots(figsize=(13, 3))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", linewidths=0.5,
                annot=True, fmt=".1f", cbar_kws={"label": "Avg Positivity Rate (%)"})
    ax.set_title(f"Average Monthly Positivity Rate (%) — {state}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("")
    fname = f"testing_heatmap_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))


# ---------------------------------------------------------------------------
# 9. GPU Outbreak Forecasting Chart
# ---------------------------------------------------------------------------

def plot_gpu_forecast(
    eval_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    state: str,
    metrics: Optional[dict] = None,
    out_dir: str = "outputs/charts",
) -> str:
    """Generate high-resolution chart of GPU outbreak forecast."""
    fig, ax = plt.subplots(figsize=(13, 6))

    # Actual historical
    ax.plot(eval_df["Date"], eval_df["ActualDailyNewCases"], color="#264653",
            label="Actual Daily Cases", linewidth=2.0, alpha=0.9)
    # Model predicted on test set
    ax.plot(eval_df["Date"], eval_df["PredictedDailyNewCases"], color="#E76F51",
            label="GPU Model Fitted (Holdout)", linewidth=2.0, linestyle="--")

    # Multi-step future forecast
    if not forecast_df.empty:
        ax.plot(forecast_df["Date"], forecast_df["ProjectedDailyNewCases"], color="#9B5DE5",
                label=f"GPU {len(forecast_df)}-Day Forward Projection", linewidth=2.5, marker="o", markersize=4)

    metrics_str = ""
    if metrics:
        metrics_str = f" [R²={metrics.get('r2', 0):.2f}, RMSE={metrics.get('rmse', 0):.0f}, MAE={metrics.get('mae', 0):.0f}]"

    ax.set_title(f"GPU Outbreak Forecasting — {state} (NVIDIA CUDA Acceleration){metrics_str}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily New Cases")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate()
    ax.legend(loc="upper left")

    fname = f"gpu_forecast_{state.lower().replace(' ', '_')}.png"
    return _save(fig, os.path.join(out_dir, fname))



# ---------------------------------------------------------------------------
# Batch chart generator
# ---------------------------------------------------------------------------

def generate_all_charts(
    cases: pd.DataFrame,
    testing: pd.DataFrame,
    vaccine: pd.DataFrame,
    state_comp: pd.DataFrame,
    monthly: pd.DataFrame,
    anomalies: pd.DataFrame,
    benchmark_summary: Optional[pd.DataFrame] = None,
    focus_state: str = "Tamil Nadu",
    rolling_col: Optional[str] = None,
    out_dir: str = "outputs/charts",
) -> List[str]:
    """Generate all charts for the focus state and save them. Returns list of paths."""
    paths = []
    tn_cases = cases[cases["State"] == focus_state]
    tn_testing = testing[testing["State"] == focus_state]
    tn_vaccine = vaccine[vaccine["State"] == focus_state]

    rc = rolling_col or ("DailyNewCases_rolling7d" if "DailyNewCases_rolling7d" in cases.columns else None)

    paths.append(plot_daily_cases(tn_cases, focus_state, rc, out_dir))
    paths.append(plot_daily_deaths(tn_cases, focus_state,
                                   "DailyNewDeaths_rolling7d" if "DailyNewDeaths_rolling7d" in tn_cases.columns else None,
                                   out_dir))
    paths.append(plot_monthly_trend(monthly, focus_state, out_dir=out_dir))
    paths.append(plot_state_comparison(state_comp, out_dir=out_dir))
    paths.append(plot_anomalies(anomalies, focus_state, out_dir=out_dir))
    paths.append(plot_vaccination(tn_vaccine, focus_state, out_dir=out_dir))
    paths.append(plot_testing_heatmap(tn_testing, focus_state, out_dir=out_dir))

    if benchmark_summary is not None and not benchmark_summary.empty:
        paths.extend(plot_benchmark(benchmark_summary, out_dir=out_dir))

    return [p for p in paths if p]   # remove empty strings
