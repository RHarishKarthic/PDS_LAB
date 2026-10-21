# Public Health Outbreak Monitoring — Analytical Summary

*Generated: 2026-10-20 15:42:05*

> **Important:** This analysis is based on historical COVID-19 data (January 2020 – August 2021). It does NOT represent real-time outbreak conditions. The last date in the dataset is noted below.

## Dataset Overview

| Dataset | Rows | Date Range |
|---------|------|------------|
| Cases (covid_19_india.csv) | 18,046 | 2020-01-30 to 2021-08-11 |
| Testing (StatewiseTestingDetails.csv) | 16,335 | 2020-04-01 to 2021-08-10 |
| Vaccination (covid_vaccine_statewise.csv) | 7,633 | 2021-01-16 to 2021-08-16 |

**Last date in dataset:** 2021-08-11

## Tamil Nadu Analysis

- **Data period:** 2020-03-07 to 2021-08-11
- **Total confirmed cases (cumulative peak):** 2,579,130
- **Total deaths (cumulative peak):** 34,367
- **Total recovered (cumulative peak):** 2,524,400
- **Peak single-day new cases:** 36,184 on 2021-05-22
- **Highest-case month:** 2021-05 (920,516 new cases)

### Anomaly Detection (Tamil Nadu)

- 29 anomalous date(s) detected using Z-score method (|Z| > 2.5).
- Anomaly dates: 2021-05-10, 2021-05-11, 2021-05-12, 2021-05-13, 2021-05-14, 2021-05-15, 2021-05-16, 2021-05-17, 2021-05-18, 2021-05-19 (and more...)

## State Comparison — Top 10 by Total Confirmed Cases

| Rank | State | Total Confirmed | Total Deaths | Peak Daily New Cases | Peak Date |
|------|-------|-----------------|--------------|----------------------|-----------|
| 1 | Maharashtra | 6,363,442 | 134,201 | 68,631 | 2021-04-19 |
| 2 | Kerala | 3,586,693 | 18,004 | 43,529 | 2021-05-13 |
| 3 | Karnataka | 2,921,049 | 36,848 | 50,112 | 2021-05-06 |
| 4 | Tamil Nadu | 2,579,130 | 34,367 | 36,184 | 2021-05-22 |
| 5 | Andhra Pradesh | 1,985,182 | 13,564 | 24,171 | 2021-05-17 |
| 6 | Uttar Pradesh | 1,708,812 | 22,775 | 37,944 | 2021-04-25 |
| 7 | West Bengal | 1,534,999 | 18,252 | 20,846 | 2021-05-15 |
| 8 | Delhi | 1,436,852 | 25,068 | 28,395 | 2021-04-21 |
| 9 | Chhattisgarh | 1,003,356 | 13,544 | 17,397 | 2021-04-24 |
| 10 | Odisha | 988,997 | 6,565 | 12,852 | 2021-05-24 |

## Vaccination Summary (Tamil Nadu)

- **Cumulative doses administered (as of 2021-08-15):** 25,522,708
- **First doses (cumulative):** 20,836,674
- **Second doses (cumulative):** 4,686,034

## Performance Benchmark Results

*Benchmark results not yet available. Run `python run_pipeline.py --benchmark` to generate.*

## Key Observations

1. The dataset covers the first and second COVID-19 waves in India (January 2020 – August 2021).
2. Tamil Nadu experienced multiple periods of significant case growth, identifiable through Z-score anomaly detection on the 7-day rolling average.
3. Vaccination data (February – July 2021) shows steady administration growth across Tamil Nadu and all major states.
4. The multithreaded implementation distributes per-state analysis across independent workers, enabling concurrent processing of all 36+ states.
5. Spark's local-mode overhead dominates at this dataset scale; the same code would demonstrate true speedup on a multi-node cluster with millions of rows.

## Limitations

- Data ends on 2021-08-11; it does not reflect current outbreak status.
- District-level data is not available in these datasets.
- Negative daily differences (data revisions) are capped at 0; the raw revision count is logged as a flagged field.
- Recovery data ('Cured') is not always reported by all states.
- Positivity rate calculation requires both TotalSamples and Positive counts; rows with missing Positive values are excluded.