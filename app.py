"""
app.py  —  Streamlit Dashboard
================================
Public Health Outbreak Monitoring and Trend Analysis
Dataset: COVID-19 in India (Kaggle — SudalaiRajkumar)
Source:  https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india

Run:  streamlit run app.py
"""

import logging
import os
import sys
import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --- Ensure src/ importable ---
sys.path.insert(0, os.path.dirname(__file__))

from src.config_loader import load_config
from src.data_loader import load_all
from src.data_cleaning import clean_all, data_quality_report
from src.validation import validate_all
from src.analysis import (
    compute_rolling_average,
    state_comparison_table,
    weekly_summary,
    monthly_summary,
    filter_state,
    get_state_list,
)
from src.anomaly_detection import rolling_zscore_anomalies, anomaly_summary

logging.basicConfig(level=logging.WARNING)

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="Public Health Outbreak Monitor — India COVID-19",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS injection
# ============================================================
st.markdown(
    """
    <style>
    .kpi-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        border: 1px solid #0f3460;
        margin-bottom: 8px;
    }
    .kpi-title { color: #a8dadc; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #e63946; font-size: 28px; font-weight: 800; margin-top: 6px; }
    .kpi-sub   { color: #8d99ae; font-size: 11px; margin-top: 4px; }
    .disclaimer {
        background: #1a1a2e; border-left: 4px solid #e63946;
        padding: 10px 16px; border-radius: 4px; color: #a8dadc; font-size: 13px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Data loading (cached)
# ============================================================
@st.cache_data(show_spinner="Loading and cleaning datasets…")
def load_data(raw_dir: str):
    cfg = load_config()
    cases_raw, testing_raw, vaccine_raw = load_all(
        raw_dir,
        cfg["data"]["cases_file"],
        cfg["data"]["testing_file"],
        cfg["data"]["vaccine_file"],
    )
    cases, testing, vaccine = clean_all(cases_raw, testing_raw, vaccine_raw)
    validate_all(cases, testing, vaccine)
    cases = compute_rolling_average(cases, "DailyNewCases", 7)
    cases = compute_rolling_average(cases, "DailyNewDeaths", 7)
    state_comp = state_comparison_table(cases)
    monthly = monthly_summary(cases)
    weekly = weekly_summary(cases)
    anomalies = rolling_zscore_anomalies(cases, z_threshold=2.5)
    dq = data_quality_report(cases, testing, vaccine)
    return cases, testing, vaccine, state_comp, monthly, weekly, anomalies, dq


cfg = load_config()
RAW_DIR = cfg["data"]["raw_dir"]

try:
    cases, testing, vaccine, state_comp, monthly, weekly, anomalies, dq = load_data(RAW_DIR)
    DATA_OK = True
except FileNotFoundError as e:
    st.error(f"❌ Dataset not found: {e}")
    st.info("Place the raw CSV files in the `data/raw/` directory and restart.")
    DATA_OK = False
    st.stop()

STATES = get_state_list(cases)
LAST_DATE = cases["Date"].max().date()
FIRST_DATE = cases["Date"].min().date()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Flag_of_India.svg", width=60)
    st.markdown("## 🦠 Outbreak Monitor")
    st.markdown("---")

    st.markdown("### 📌 Filters")
    selected_state = st.selectbox("State / Union Territory", STATES,
                                  index=STATES.index("Tamil Nadu") if "Tamil Nadu" in STATES else 0)

    date_range = st.date_input(
        "Date Range",
        value=[FIRST_DATE, LAST_DATE],
        min_value=FIRST_DATE,
        max_value=LAST_DATE,
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        d_start, d_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    else:
        d_start, d_end = pd.Timestamp(FIRST_DATE), pd.Timestamp(LAST_DATE)

    z_threshold = st.slider("Anomaly Z-Score Threshold", 1.5, 4.0, 2.5, 0.1)
    rolling_win = st.slider("Rolling Average Window (days)", 3, 14, 7)

    st.markdown("---")
    st.markdown("### ℹ️ Dataset Info")
    st.markdown(
        f"""
        **Source:** [Kaggle — covid19-in-india](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india)  
        **Period:** {FIRST_DATE} → {LAST_DATE}  
        **States:** {len(STATES)}  
        """
    )

# ============================================================
# Filter data for selected state + date range
# ============================================================
state_df = filter_state(cases, selected_state)
state_df = state_df[(state_df["Date"] >= d_start) & (state_df["Date"] <= d_end)].copy()

if not state_df.empty:
    state_df = compute_rolling_average(state_df, "DailyNewCases", rolling_win, group_col="State")
    state_df = compute_rolling_average(state_df, "DailyNewDeaths", rolling_win, group_col="State")
    state_anom = rolling_zscore_anomalies(
        state_df,
        rolling_window=rolling_win,
        z_threshold=z_threshold,
    )
else:
    state_anom = pd.DataFrame()

testing_state = filter_state(testing, selected_state)
testing_state = testing_state[(testing_state["Date"] >= d_start) & (testing_state["Date"] <= d_end)]

vaccine_state = filter_state(vaccine, selected_state)
vaccine_state = vaccine_state[(vaccine_state["Date"] >= d_start) & (vaccine_state["Date"] <= d_end)]

# ============================================================
# HEADER
# ============================================================
st.markdown(
    "# 🦠 Public Health Outbreak Monitoring & Trend Analysis",
)
st.markdown(
    f"**COVID-19 in India** · Historical data ({FIRST_DATE} – {LAST_DATE}) · "
    f"Source: [Kaggle/SudalaiRajkumar](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india)"
)

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview", "📈 Trends", "🗺️ State Comparison",
    "⚠️ Anomalies", "💉 Vaccination", "🧪 Testing",
    "⚡ Benchmarks & Lab Concepts"
])

# ============================================================
# TAB 1: OVERVIEW — KPI CARDS
# ============================================================
with tab1:
    st.subheader(f"📊 {selected_state} — Key Indicators")

    if state_df.empty:
        st.warning("No data available for the selected state and date range.")
    else:
        total_confirmed = int(cases[cases["State"] == selected_state]["Confirmed"].max())
        total_deaths = int(cases[cases["State"] == selected_state]["Deaths"].max())
        total_recovered = int(cases[cases["State"] == selected_state]["Cured"].max())
        peak_day = int(state_df["DailyNewCases"].max())
        peak_date = state_df.loc[state_df["DailyNewCases"].idxmax(), "Date"].date()

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, title, val, sub in [
            (c1, "Total Confirmed", f"{total_confirmed:,}", "Cumulative peak"),
            (c2, "Total Deaths", f"{total_deaths:,}", "Cumulative peak"),
            (c3, "Total Recovered", f"{total_recovered:,}", "Cumulative peak"),
            (c4, "Peak Single Day", f"{peak_day:,}", f"on {peak_date}"),
            (c5, "Data Coverage", f"{len(state_df)} days", f"{d_start.date()} – {d_end.date()}"),
        ]:
            col.markdown(
                f'<div class="kpi-card"><div class="kpi-title">{title}</div>'
                f'<div class="kpi-value">{val}</div><div class="kpi-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Quick overview chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=state_df["Date"], y=state_df["DailyNewCases"],
            name="Daily New Cases", marker_color="rgba(230,57,70,0.5)"
        ))
        rc_col = f"DailyNewCases_rolling{rolling_win}d"
        if rc_col in state_df.columns:
            fig.add_trace(go.Scatter(
                x=state_df["Date"], y=state_df[rc_col],
                name=f"{rolling_win}-Day Rolling Avg",
                line=dict(color="#e63946", width=2.5)
            ))
        fig.update_layout(
            title=f"Daily New Cases — {selected_state}",
            xaxis_title="Date", yaxis_title="Cases",
            template="plotly_dark", hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
            height=420,
        )
        st.plotly_chart(fig, width='stretch')

    # Data Quality Summary
    st.markdown("---")
    st.subheader("📋 Data Quality Summary")
    st.dataframe(dq, width='stretch')


# ============================================================
# TAB 2: TRENDS
# ============================================================
with tab2:
    st.subheader(f"📈 Trends — {selected_state}")

    if state_df.empty:
        st.warning("No data for selected filters.")
    else:
        # Monthly trend
        monthly_state = monthly[
            (monthly["State"] == selected_state)
        ].copy()
        monthly_state = monthly_state[
            monthly_state["MonthStart"].between(d_start, d_end)
        ]

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Monthly New Cases**")
            fig_m = px.bar(
                monthly_state, x="YearMonth", y="DailyNewCases",
                labels={"DailyNewCases": "Total New Cases", "YearMonth": "Month"},
                color_discrete_sequence=["#e63946"],
                template="plotly_dark",
            )
            fig_m.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig_m, width='stretch')

        with col_b:
            st.markdown("**Monthly Deaths**")
            fig_d = px.bar(
                monthly_state, x="YearMonth", y="DailyNewDeaths",
                labels={"DailyNewDeaths": "Total Deaths", "YearMonth": "Month"},
                color_discrete_sequence=["#6d6875"],
                template="plotly_dark",
            )
            fig_d.update_layout(height=350, xaxis_tickangle=-45)
            st.plotly_chart(fig_d, width='stretch')

        st.markdown("---")

        # Daily new recoveries over time
        st.markdown("**Daily New Recoveries**")
        rec_col = "DailyNewRecoveries" if "DailyNewRecoveries" in state_df.columns else "Cured"
        rec_label = "Daily New Recoveries" if rec_col == "DailyNewRecoveries" else "Cumulative Recovered"
        fig_r = px.area(
            state_df, x="Date", y=rec_col,
            labels={rec_col: rec_label},
            color_discrete_sequence=["#2a9d8f"],
            template="plotly_dark",
        )
        fig_r.update_layout(height=350)
        st.plotly_chart(fig_r, width='stretch')

        # Table
        st.markdown("**Processed Data (filtered)**")
        show_cols = ["Date", "DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries",
                     "Confirmed", "Deaths", "Cured"]
        show_cols = [c for c in show_cols if c in state_df.columns]
        st.dataframe(state_df[show_cols].tail(100), width='stretch')

        # Download
        buf = io.BytesIO()
        state_df.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Download filtered data (CSV)",
            data=buf.getvalue(),
            file_name=f"{selected_state.replace(' ', '_')}_filtered.csv",
            mime="text/csv",
        )


# ============================================================
# TAB 3: STATE COMPARISON
# ============================================================
with tab3:
    st.subheader("🗺️ All-India State Comparison")
    top_n = st.slider("Show top N states", 5, 36, 15)
    comp_df = state_comp.head(top_n)

    fig_comp = px.bar(
        comp_df[::-1],
        x="TotalConfirmed", y="State",
        orientation="h",
        color="TotalConfirmed",
        color_continuous_scale="Reds",
        labels={"TotalConfirmed": "Total Confirmed (Cumulative Peak)", "State": "State"},
        title=f"Top {top_n} States — Total Confirmed COVID-19 Cases",
        template="plotly_dark",
    )
    fig_comp.update_layout(height=600, coloraxis_showscale=False)
    st.plotly_chart(fig_comp, width='stretch')

    st.markdown("**Peak Daily New Cases per State**")
    fig_peak = px.scatter(
        comp_df,
        x="TotalConfirmed", y="PeakDailyNewCases",
        text="State", size="TotalDeaths",
        color="TotalDeaths", color_continuous_scale="Purples",
        labels={
            "TotalConfirmed": "Total Confirmed",
            "PeakDailyNewCases": "Peak Daily New Cases",
            "TotalDeaths": "Total Deaths",
        },
        title="Confirmed Cases vs Peak Daily New Cases (bubble size = total deaths)",
        template="plotly_dark",
    )
    fig_peak.update_traces(textposition="top center", textfont_size=9)
    fig_peak.update_layout(height=500)
    st.plotly_chart(fig_peak, width='stretch')

    st.markdown("**State Comparison Table**")
    st.dataframe(state_comp, width='stretch')

    buf2 = io.BytesIO()
    state_comp.to_csv(buf2, index=False)
    st.download_button(
        "⬇️ Download state comparison (CSV)",
        data=buf2.getvalue(),
        file_name="state_comparison.csv",
        mime="text/csv",
    )


# ============================================================
# TAB 4: ANOMALY DETECTION
# ============================================================
with tab4:
    st.subheader(f"⚠️ Anomaly Detection — {selected_state}")
    st.markdown(
        f"""
        **Method:** Rolling Z-Score on {rolling_win}-day average of Daily New Cases  
        **Threshold:** |Z| > {z_threshold:.1f}  
        **What it detects:** Dates where the rolling case count deviated significantly 
        from the state's own historical mean — not absolute case count thresholds.  
        **Limitations:** Statistical flag only; does NOT diagnose an outbreak cause.
        """
    )

    if state_anom.empty:
        st.warning("No anomaly data available for selected filters.")
    else:
        anom_col = f"Rolling{rolling_win}d_DailyNewCases"
        rc_present = anom_col in state_anom.columns

        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(
            x=state_anom["Date"], y=state_anom["DailyNewCases"],
            name="Daily New Cases", marker_color="rgba(230,57,70,0.35)"
        ))
        if rc_present:
            fig_a.add_trace(go.Scatter(
                x=state_anom["Date"], y=state_anom[anom_col],
                name=f"{rolling_win}-Day Rolling Avg",
                line=dict(color="#E9C46A", width=2)
            ))
        # Anomaly markers
        anom_pts = state_anom[state_anom["IsAnomaly"]]
        n_anom = len(anom_pts)
        fig_a.add_trace(go.Scatter(
            x=anom_pts["Date"],
            y=anom_pts["DailyNewCases"],
            mode="markers",
            name=f"⚠️ Anomaly ({n_anom} dates)",
            marker=dict(color="#FF6B6B", size=10, symbol="triangle-up",
                        line=dict(color="white", width=1)),
        ))
        fig_a.update_layout(
            title=f"Anomaly Detection — {selected_state} (|Z| > {z_threshold:.1f}, {rolling_win}-day window: {n_anom} dates)",
            xaxis_title="Date", yaxis_title="Daily New Cases",
            template="plotly_dark", hovermode="x unified", height=430,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_a, width='stretch')

        # Z-Score subplot
        fig_z = px.line(
            state_anom, x="Date", y="ZScore",
            title=f"Z-Score Time Series ({rolling_win}-day window, Threshold ±{z_threshold:.1f})",
            template="plotly_dark",
            labels={"ZScore": "Z-Score"},
        )
        fig_z.add_hline(y=z_threshold, line_dash="dash", line_color="red",
                        annotation_text=f"+{z_threshold:.1f}")
        fig_z.add_hline(y=-z_threshold, line_dash="dash", line_color="red",
                        annotation_text=f"-{z_threshold:.1f}")
        fig_z.update_layout(height=300)
        st.plotly_chart(fig_z, width='stretch')

        n_anom = len(anom_pts)
        st.metric("Anomalous Dates Detected", n_anom)

        if not anom_pts.empty:
            st.markdown("**Anomalous Dates**")
            st.dataframe(
                anom_pts[["Date", "DailyNewCases", "ZScore", "AnomalyDirection"]].sort_values("Date"),
                width='stretch',
            )


# ============================================================
# TAB 5: VACCINATION
# ============================================================
with tab5:
    st.subheader(f"💉 Vaccination Data — {selected_state}")
    if vaccine_state.empty:
        st.warning(f"No vaccination data available for '{selected_state}' in the selected date range. "
                   "Vaccination data covers February–July 2021.")
    else:
        fig_v = go.Figure()
        if "DailyNewDoses" in vaccine_state.columns:
            fig_v.add_trace(go.Bar(
                x=vaccine_state["Date"], y=vaccine_state["DailyNewDoses"],
                name="Daily New Doses", marker_color="rgba(233,196,106,0.6)", yaxis="y"
            ))
        if "Total Doses Administered" in vaccine_state.columns:
            fig_v.add_trace(go.Scatter(
                x=vaccine_state["Date"], y=vaccine_state["Total Doses Administered"],
                name="Cumulative Doses", line=dict(color="#E76F51", width=2.5), yaxis="y2"
            ))
        fig_v.update_layout(
            title=f"Vaccination Progress — {selected_state}",
            xaxis_title="Date",
            yaxis=dict(title="Daily New Doses"),
            yaxis2=dict(title="Cumulative Doses", overlaying="y", side="right"),
            template="plotly_dark", hovermode="x unified", height=420,
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig_v, width='stretch')

        # Dose breakdown
        dose_cols = [c for c in ["First Dose Administered", "Second Dose Administered"] if c in vaccine_state.columns]
        if dose_cols and not vaccine_state.empty and vaccine_state["Date"].notna().any():
            last_row = vaccine_state.loc[vaccine_state["Date"].idxmax()]
            c1, c2 = st.columns(2)
            first_val = last_row.get("First Dose Administered", 0)
            second_val = last_row.get("Second Dose Administered", 0)
            f_str = f"{int(first_val):,}" if pd.notna(first_val) else "N/A"
            s_str = f"{int(second_val):,}" if pd.notna(second_val) else "N/A"
            c1.metric("First Doses (Cumulative)", f_str)
            c2.metric("Second Doses (Cumulative)", s_str)

        st.dataframe(vaccine_state[["Date"] + [c for c in vaccine_state.columns if c != "Date"]].tail(60),
                     width='stretch')


# ============================================================
# TAB 6: TESTING
# ============================================================
with tab6:
    st.subheader(f"🧪 Testing Data — {selected_state}")
    if testing_state.empty:
        st.warning(f"No testing data available for '{selected_state}' in the selected range.")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_t = px.bar(
                testing_state, x="Date", y="DailyNewSamples",
                title="Daily New Samples Tested",
                color_discrete_sequence=["#A8DADC"],
                template="plotly_dark",
                labels={"DailyNewSamples": "Samples Tested"},
            )
            fig_t.update_layout(height=350)
            st.plotly_chart(fig_t, width='stretch')

        with col_t2:
            if "PositivityRate" in testing_state.columns and testing_state["PositivityRate"].notna().any():
                fig_pos = px.line(
                    testing_state.dropna(subset=["PositivityRate"]),
                    x="Date", y="PositivityRate",
                    title="Positivity Rate (%)",
                    color_discrete_sequence=["#E9C46A"],
                    template="plotly_dark",
                    labels={"PositivityRate": "Positivity Rate (%)"},
                )
                fig_pos.add_hline(y=5, line_dash="dash", line_color="red",
                                  annotation_text="WHO 5% threshold")
                fig_pos.update_layout(height=350)
                st.plotly_chart(fig_pos, width='stretch')
            else:
                st.info("Positivity rate not available (Positive column has too many missing values).")

        test_display_cols = [c for c in ["Date", "State", "TotalSamples", "Positive",
                                         "Negative", "DailyNewSamples", "PositivityRate"]
                             if c in testing_state.columns]
        st.dataframe(
            testing_state[test_display_cols].tail(60),
            width='stretch',
        )


# ============================================================
# TAB 7: BENCHMARKS & LAB CONCEPTS
# ============================================================
with tab7:
    st.subheader("⚡ Laboratory Concepts & Performance Benchmarking")

    st.markdown(
        """
## Lab Concept 1: Apache Spark (PySpark)

**What is Spark?**  
Apache Spark is a distributed, in-memory computing framework. Data is represented as 
a *Spark DataFrame* — a logically partitioned, schema-aware collection processed across 
nodes using a DAG (Directed Acyclic Graph) execution model.

**How it's used in this project:**
- The raw CSV is loaded into a Spark DataFrame via `SparkSession.read.csv`.
- Transformations: `filter()`, `groupBy()`, `agg()`, `withColumn()`, `orderBy()`.
- Window functions: 7-day rolling average using `pyspark.sql.window.Window`.
- Spark SQL: temp view `covid_cases` queried with `.sql("SELECT ... GROUP BY ...")`.
- Results collected with `.toPandas()` for downstream dashboard use.

**Justification for Spark:**  
Real public health outbreak monitoring ingests millions of records from hospitals, labs, 
and contact-tracing apps simultaneously. Spark's architecture can process such streams 
in near-real-time across a cluster of machines. In this project we run in `local[*]` mode 
(single machine, all CPU cores) to demonstrate the programming pattern that scales to 
a cluster by changing only the `master` config string.

**Local-mode caveat:**  
On ~18 k rows, Spark's JVM startup and serialisation overhead **exceeds** its benefit.  
This is expected, documented, and confirmed in the benchmarks below.

---

## Lab Concept 2: Python Multithreading

**What is multithreading?**  
Multithreading allows multiple threads to run concurrently within a single Python process, 
sharing memory. Python's `concurrent.futures.ThreadPoolExecutor` provides a clean API.

**How it's used in this project:**  
Each state's analysis (rolling average + anomaly detection + weekly summary) is dispatched 
as an independent `Future` to the thread pool. The tasks are embarrassingly independent — 
Tamil Nadu's rolling average cannot affect Maharashtra's result.

**GIL transparency:**  
Python's Global Interpreter Lock (GIL) means CPU-bound Python bytecode does not truly 
run in parallel. However: (a) pandas releases the GIL during some NumPy operations, 
(b) I/O-bound extensions (reading per-state files) would show genuine speedup, and 
(c) the pattern is directly reusable with `ProcessPoolExecutor` for full CPU parallelism.

---
"""
    )

    # Load benchmark results if available
    bench_path = os.path.join(cfg["outputs"]["benchmarks_dir"], "benchmark_summary.csv")
    if os.path.exists(bench_path):
        bdf = pd.read_csv(bench_path)
        st.markdown("### 📊 Benchmark Results")
        st.dataframe(bdf, width='stretch')

        # Execution time chart
        bdf_clean = bdf.dropna(subset=["MedianElapsedSeconds"])
        if not bdf_clean.empty:
            fig_b = px.bar(
                bdf_clean, x="Implementation", y="MedianElapsedSeconds",
                color="Implementation",
                color_discrete_map={
                    "Sequential (pandas)": "#264653",
                    "Multithreaded (4 workers)": "#2A9D8F",
                    "Spark (local[*])": "#E9C46A",
                },
                title="Median Execution Time Comparison (3 runs)",
                labels={"MedianElapsedSeconds": "Median Elapsed Time (s)"},
                template="plotly_dark",
                text="MedianElapsedSeconds",
            )
            fig_b.update_traces(texttemplate="%{text:.2f}s", textposition="outside")
            fig_b.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_b, width='stretch')

            # Speedup chart
            speed_df = bdf_clean.dropna(subset=["SpeedupVsSequential"])
            if not speed_df.empty:
                fig_sp = px.bar(
                    speed_df, x="Implementation", y="SpeedupVsSequential",
                    title="Speedup vs Sequential Baseline",
                    labels={"SpeedupVsSequential": "Speedup Factor (×)"},
                    template="plotly_dark",
                    text="SpeedupVsSequential",
                    color_discrete_sequence=["#2A9D8F"],
                )
                fig_sp.add_hline(y=1.0, line_dash="dash", line_color="red",
                                 annotation_text="Baseline = 1×")
                fig_sp.update_traces(texttemplate="%{text:.3f}×", textposition="outside")
                fig_sp.update_layout(showlegend=False, height=380)
                st.plotly_chart(fig_sp, width='stretch')

        st.markdown(
            "> **Formula:** Speedup = T_sequential / T_parallel  \n"
            "> **Efficiency** = Speedup / n_workers (for threaded only)  \n"
            "> Spark overhead on this small dataset is expected and documented."
        )
    else:
        st.info(
            "Benchmark results not yet generated. Run the pipeline with:\n\n"
            "```bash\n"
            "python run_pipeline.py --benchmark --spark\n"
            "```"
        )

    # Runs detail
    runs_path = os.path.join(cfg["outputs"]["benchmarks_dir"], "benchmark_runs.csv")
    if os.path.exists(runs_path):
        rdf = pd.read_csv(runs_path)
        st.markdown("### Per-Run Details")
        st.dataframe(rdf, width='stretch')


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(
    "**Public Health Outbreak Monitoring and Trend Analysis**  \n"
    "Dataset: [COVID-19 in India — Kaggle/SudalaiRajkumar]"
    "(https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india)  \n"
    "Lab Concepts: Apache Spark (PySpark) · Python Multithreading"
)
