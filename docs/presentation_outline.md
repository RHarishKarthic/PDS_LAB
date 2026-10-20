# Presentation Outline
## Public Health Outbreak Monitoring and Trend Analysis
### (12–15 Slides)

---

### Slide 1 — Title Slide
**Project Title:** Public Health Outbreak Monitoring and Trend Analysis  
**Dataset:** COVID-19 in India (Kaggle)  
**Lab Concepts:** Apache Spark · Python Multithreading  
- Student Name: [YOUR NAME]
- Register Number: [YOUR REG. NO.]
- Department: [YOUR DEPARTMENT]
- Guide: [FACULTY NAME]
- Date: [DATE]

---

### Slide 2 — Problem Statement

**Why does this matter?**
- Outbreak monitoring requires processing millions of health records from multiple states simultaneously.
- Manual or sequential analysis is too slow to support timely public health decisions.
- Anomaly detection must be automated, transparent, and reproducible.

**Specific Problem:**
- Process COVID-19 data for 42 Indian states/UTs.
- Identify periods of unusual case growth.
- Compare the performance of sequential vs parallel computing approaches.

---

### Slide 3 — Project Objectives

1. Build a complete public health data pipeline (load → clean → validate → analyse → visualise).
2. Demonstrate **Apache Spark** (PySpark) for distributed data processing.
3. Demonstrate **Python Multithreading** for concurrent state-level analysis.
4. Detect anomalous surges using Z-score method.
5. Build an interactive Streamlit + Plotly dashboard.
6. Benchmark all three implementations with real execution times.
7. Validate with 68 automated pytest tests.

---

### Slide 4 — Dataset and Source

| Dataset | File | Rows | Date Range |
|---------|------|------|------------|
| State-wise Cases | covid_19_india.csv | 18,110 | 2020-01-30 → 2021-08-11 |
| State-wise Testing | StatewiseTestingDetails.csv | 16,336 | 2020-04-01 → 2021-08-10 |
| State-wise Vaccination | covid_vaccine_statewise.csv | 7,845 | 2021-01-16 → 2021-08-16 |

**Source:** [Kaggle — covid19-in-india by SudalaiRajkumar](https://www.kaggle.com/datasets/sudalairajkumar/covid19-in-india)

**Important:**
- ⚠️ Historical data only — NOT real-time.
- Last date in dataset: **2021-08-11**.
- All case counts are **cumulative daily snapshots**.

---

### Slide 5 — System Architecture

```
Raw CSVs → data_loader → data_cleaning → validation → cleaned CSVs
                                                          ↓
                              ┌───────────────────────────┤
                              │                           │
                    Sequential (pandas)    Multithreaded (4 workers)
                              │                           │
                              └───────────┬───────────────┘
                                          │
                                     Spark (PySpark)
                                          │
                              Analysis → Visualisation → Dashboard
```

**Modules:** config_loader, data_loader, data_cleaning, validation, analysis, sequential_processor, threaded_processor, spark_processor, anomaly_detection, benchmarking, visualization, reporting, app.py

---

### Slide 6 — Data Preprocessing

**Key Cleaning Steps:**
1. **State alias resolution**: "Bihar****" → "Bihar"; "Cases being reassigned" → dropped.
2. **Cumulative → daily**: `DailyNewCases = diff(Confirmed)` per state per date.
3. **Negative diff capping**: data-revision artefacts capped at 0, flagged for audit.
4. **Duplicate removal**: 1 duplicate testing row removed.
5. **India aggregate removed**: from vaccine dataset (would double-count all states).
6. **Temporal columns**: Year, Month, Week, YearMonth, YearWeek added.

**Data Quality:**
- 63 unassignable-state rows dropped from cases.
- 212 'India' aggregate rows removed from vaccine.
- All validations pass: non-negative daily counts, correct dtypes, no nulls in key columns.

---

### Slide 7 — Lab Concept 1: Apache Spark

**What is Spark?**
- Distributed, in-memory data processing framework.
- DAG-based execution — lazy transformations, triggered by actions.
- Same code scales from a laptop to a 1000-node cluster.

**How Used in This Project:**
```python
# Spark SQL
spark.sql("SELECT State, MAX(Confirmed) AS Total FROM covid_cases GROUP BY State")

# Window function — 7-day rolling average
Window.partitionBy("State").orderBy(F.col("Date").cast("long")).rowsBetween(-6, 0)
df = df.withColumn("RollingAvg7d", F.avg("DailyNewCases").over(window_rolling))
```

**Operations:** `filter`, `groupBy().agg()`, `withColumn()`, `Window`, Spark SQL, `.toPandas()`

---

### Slide 8 — Lab Concept 2: Python Multithreading

**What is Multithreading?**
- Multiple threads run concurrently within one Python process.
- `concurrent.futures.ThreadPoolExecutor` — high-level API.
- GIL limits CPU-bound parallelism; genuine speedup for I/O-bound tasks.

**How Used:**
```python
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(_analyse_one_state, cases, state): state
        for state in all_states   # 42 states
    }
    for future in as_completed(futures):
        result = future.result()  # rolling avg + anomalies + weekly summary
```

**Why suitable here:** Tasks are embarrassingly independent (Tamil Nadu's analysis cannot affect Maharashtra's).

---

### Slide 9 — Dashboard Screenshots

*[Insert screenshot of Dashboard Tab 1 — Overview with KPI cards]*
*[Insert screenshot of Tab 2 — Trends with monthly bar charts]*
*[Insert screenshot of Tab 4 — Anomaly Detection with Z-score panel]*
*[Insert screenshot of Tab 7 — Benchmarks with execution time chart]*

**Run the dashboard:**
```bash
streamlit run app.py
```

---

### Slide 10 — Analytical Results: Tamil Nadu

| Metric | Value |
|--------|-------|
| Total Confirmed (cumulative peak) | 2,674,047 |
| Total Deaths (cumulative peak) | 35,399 |
| Total Recovered (cumulative peak) | 2,614,068 |
| Peak Single-Day New Cases | 36,184 (May 12, 2021) |
| Highest Case Month | May 2021 |
| Anomalous Dates (Z > 2.5) | 29 dates |

*[Insert daily_cases_tamil_nadu.png chart here]*

---

### Slide 11 — Geographic Comparison

*[Insert state_comparison_top15.png chart here]*

**Top 5 States (Total Confirmed):**
1. Maharashtra — 6,358,491
2. Kerala — 3,478,419
3. Karnataka — 2,921,661
4. Tamil Nadu — 2,674,047
5. Andhra Pradesh — 1,963,250

---

### Slide 12 — Performance Benchmark

| Implementation | Median Time (s) | Speedup |
|----------------|----------------|---------|
| Sequential (pandas) | 0.738 | 1.000× |
| Multithreaded (4 workers) | 1.581 | 0.467× |
| Spark local[*] | ~25–35 s | <0.1× |

*[Insert benchmark_execution_time.png here]*

**Explanation:**
- Multithreading overhead exceeds benefit on small data (GIL + thread sync).
- Spark JVM startup dominates on 18 k rows.
- At scale (millions of rows, cluster), both would outperform sequential.

---

### Slide 13 — Technology Justification

| Technology | Why Chosen |
|------------|-----------|
| Apache Spark | Industry-standard for large-scale health data; scales to cluster; SQL support |
| Python Multithreading | Embarrassingly parallel state tasks; simple API; extensible to ProcessPool |
| Streamlit | Rapid prototyping; Python-native; no frontend code required |
| Plotly | Interactive charts; hover/zoom/filter; dark theme |
| pandas | Numerical backbone; first-class datetime/groupby support |
| pytest | Automated correctness validation; synthetic fixtures; no internet needed |

---

### Slide 14 — Limitations

1. **Dataset ends 2021-08-11** — historical data only, not real-time.
2. **State-level only** — no district granularity.
3. **GIL limits threading speedup** for CPU-bound tasks.
4. **Spark local-mode overhead** exceeds benefit on small datasets.
5. **Negative daily differences** (data revisions) capped at 0.
6. **Windows hostname restriction** requires Spark IP override (`SPARK_LOCAL_IP=127.0.0.1`).

---

### Slide 15 — Conclusion and Future Scope

**Accomplished:**
✅ Complete 9-stage pipeline (load → clean → validate → analyse → visualise → benchmark → report)  
✅ Two lab concepts implemented and executed (Spark + Multithreading)  
✅ 68/68 tests passing  
✅ Real benchmark measurements with honest analysis  
✅ Interactive Streamlit dashboard  

**Future Scope:**
- Integrate Apache Kafka for real-time streaming data.
- Add district-level data when available.
- Implement time-series forecasting (ARIMA / Prophet).
- Deploy on a multi-node Spark cluster for true distributed speedup.
- Add population data for incidence-rate normalisation.
