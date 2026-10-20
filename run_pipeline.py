"""
run_pipeline.py
---------------
Main entry point for the data processing pipeline.

Usage
-----
    python run_pipeline.py                  # full pipeline (no benchmark, no Spark)
    python run_pipeline.py --benchmark      # includes benchmarking
    python run_pipeline.py --spark          # includes Spark analysis
    python run_pipeline.py --all            # benchmark + Spark
    python run_pipeline.py --state "Kerala" # change focus state
"""

import argparse
import logging
import os
import sys

# Ensure src/ is importable when running from the project root
sys.path.insert(0, os.path.dirname(__file__))


def setup_logging(cfg: dict) -> None:
    log_path = cfg.get("logging", {}).get("file", "outputs/reports/pipeline.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    level_str = cfg.get("logging", {}).get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Public Health Outbreak Monitoring Pipeline"
    )
    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run performance benchmarks")
    parser.add_argument("--spark", action="store_true",
                        help="Run Spark analysis")
    parser.add_argument("--gpu", action="store_true",
                        help="Run GPU-accelerated outbreak modeling and forecasting")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run benchmark + Spark + GPU")
    parser.add_argument("--state", default=None,
                        help="Override focus state (default: Tamil Nadu)")
    args = parser.parse_args()

    # --- Load config ---
    from src.config_loader import load_config
    cfg = load_config(args.config)

    setup_logging(cfg)
    logger = logging.getLogger("pipeline")
    logger.info("=" * 60)
    logger.info("Public Health Outbreak Monitoring Pipeline — START")
    logger.info("=" * 60)

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]
    focus_state = args.state or cfg["analysis"]["focus_state"]
    rolling_window = cfg["analysis"]["rolling_window"]
    z_threshold = cfg["analysis"]["anomaly_zscore_threshold"]
    max_workers = cfg["threading"]["max_workers"]
    charts_dir = cfg["outputs"]["charts_dir"]
    out_reports = cfg["outputs"]["reports_dir"]
    benchmarks_dir = cfg["outputs"]["benchmarks_dir"]
    bench_runs = cfg["benchmarking"]["runs"]

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)
    os.makedirs(benchmarks_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 1: Load raw data                                               #
    # ------------------------------------------------------------------ #
    logger.info("Stage 1: Loading datasets from '%s'", raw_dir)
    from src.data_loader import load_all
    cases_raw, testing_raw, vaccine_raw = load_all(
        raw_dir,
        cfg["data"]["cases_file"],
        cfg["data"]["testing_file"],
        cfg["data"]["vaccine_file"],
    )

    # ------------------------------------------------------------------ #
    # Stage 2: Clean                                                       #
    # ------------------------------------------------------------------ #
    logger.info("Stage 2: Cleaning datasets")
    from src.data_cleaning import clean_all, data_quality_report
    cases, testing, vaccine = clean_all(cases_raw, testing_raw, vaccine_raw)

    # Data quality report
    dq = data_quality_report(cases, testing, vaccine)
    dq_path = os.path.join(out_reports, "data_quality.csv")
    dq.to_csv(dq_path, index=False)
    logger.info("Data quality report saved to %s", dq_path)

    # ------------------------------------------------------------------ #
    # Stage 3: Validate                                                    #
    # ------------------------------------------------------------------ #
    logger.info("Stage 3: Validating datasets")
    from src.validation import validate_all
    validate_all(cases, testing, vaccine)

    # ------------------------------------------------------------------ #
    # Stage 4: Save cleaned datasets                                       #
    # ------------------------------------------------------------------ #
    logger.info("Stage 4: Saving cleaned datasets")
    cases.to_csv(cfg["outputs"]["cleaned_cases"], index=False)
    testing.to_csv(cfg["outputs"]["cleaned_testing"], index=False)
    vaccine.to_csv(cfg["outputs"]["cleaned_vaccine"], index=False)
    logger.info("Cleaned data saved to '%s'", processed_dir)

    # ------------------------------------------------------------------ #
    # Stage 5: Analysis                                                    #
    # ------------------------------------------------------------------ #
    logger.info("Stage 5: Running analysis (focus state: %s)", focus_state)
    from src.analysis import (
        tn_detailed_analysis,
        compute_rolling_average,
        monthly_summary,
    )
    from src.anomaly_detection import rolling_zscore_anomalies

    results = tn_detailed_analysis(
        cases, testing, vaccine, rolling_window, z_threshold
    )
    cases_tn = results["cases_tn"]
    testing_tn = results["testing_tn"]
    vaccine_tn = results["vaccine_tn"]
    weekly_tn = results["weekly_tn"]
    monthly_tn = results["monthly_tn"]
    state_comp = results["state_comparison"]

    # Anomaly detection (full anomaly_detection module)
    anomalies_tn = rolling_zscore_anomalies(
        cases_tn, z_threshold=z_threshold
    )

    # Save analysis results
    state_comp.to_csv(os.path.join(out_reports, "state_comparison.csv"), index=False)
    weekly_tn.to_csv(os.path.join(out_reports, "weekly_tn.csv"), index=False)
    monthly_tn.to_csv(os.path.join(out_reports, "monthly_tn.csv"), index=False)
    anomalies_tn.to_csv(os.path.join(out_reports, "anomalies_tn.csv"), index=False)
    logger.info("Analysis results saved to '%s'", out_reports)

    # ------------------------------------------------------------------ #
    # Stage 6: Visualizations                                              #
    # ------------------------------------------------------------------ #
    logger.info("Stage 6: Generating charts")
    from src.visualization import generate_all_charts

    # Compute rolling for all states for the charts
    cases_all = compute_rolling_average(cases, "DailyNewCases", rolling_window)
    cases_all = compute_rolling_average(cases_all, "DailyNewDeaths", rolling_window)
    monthly_all = monthly_summary(cases)
    anomalies_all = rolling_zscore_anomalies(cases_all, z_threshold=z_threshold)

    chart_paths = generate_all_charts(
        cases=cases_all,
        testing=testing,
        vaccine=vaccine,
        state_comp=state_comp,
        monthly=monthly_all,
        anomalies=anomalies_all,
        benchmark_summary=None,   # will be updated after benchmark
        focus_state=focus_state,
        rolling_col="DailyNewCases_rolling7d",
        out_dir=charts_dir,
    )
    logger.info("Generated %d charts in '%s'", len(chart_paths), charts_dir)

    # ------------------------------------------------------------------ #
    # Stage 7: Spark (optional)                                            #
    # ------------------------------------------------------------------ #
    spark_result = None
    if args.spark or args.run_all:
        logger.info("Stage 7: Running Spark analysis")
        try:
            from src.spark_processor import run_spark, stop_spark
            spark_result = run_spark(
                raw_dir=raw_dir,
                app_name=cfg["spark"]["app_name"],
                master=cfg["spark"]["master"],
                driver_memory=cfg["spark"]["driver_memory"],
                log_level=cfg["spark"]["log_level"],
                rolling_window=rolling_window,
            )
            spark_result["state_totals"].to_csv(
                os.path.join(out_reports, "spark_state_totals.csv"), index=False
            )
            spark_result["monthly_totals"].to_csv(
                os.path.join(out_reports, "spark_monthly_totals.csv"), index=False
            )
            logger.info(
                "Spark analysis done in %.2f s", spark_result["elapsed_seconds"]
            )
            stop_spark()
        except Exception as e:
            logger.error("Spark analysis failed: %s", e)

    # ------------------------------------------------------------------ #
    # Stage 7.5: GPU Outbreak Forecasting (optional / --all)              #
    # ------------------------------------------------------------------ #
    gpu_result = None
    if args.gpu or args.run_all:
        logger.info("Stage 7.5: Running GPU Outbreak Modeling & Forecasting (Focus: %s)", focus_state)
        try:
            from src.gpu_processor import run_gpu
            from src.visualization import plot_gpu_forecast
            gpu_result = run_gpu(raw_dir=raw_dir, rolling_window=rolling_window, focus_state=focus_state)
            fc_data = gpu_result["forecasting"]
            fc_csv = os.path.join(out_reports, f"gpu_forecast_{focus_state.lower().replace(' ', '_')}.csv")
            fc_data["forecast_df"].to_csv(fc_csv, index=False)
            eval_csv = os.path.join(out_reports, f"gpu_eval_{focus_state.lower().replace(' ', '_')}.csv")
            fc_data["evaluation_df"].to_csv(eval_csv, index=False)
            plot_gpu_forecast(
                fc_data["evaluation_df"],
                fc_data["forecast_df"],
                focus_state,
                metrics=fc_data["metrics"],
                out_dir=charts_dir,
            )
            logger.info(
                "GPU modeling completed in %.2f s (Device: %s)",
                gpu_result["elapsed_seconds"], gpu_result["target_device"].upper()
            )
        except Exception as e:
            logger.error("GPU modeling failed: %s", e)

    # ------------------------------------------------------------------ #
    # Stage 8: Benchmarking (optional)                                     #
    # ------------------------------------------------------------------ #
    benchmark_summary = None
    if args.benchmark or args.run_all:
        logger.info("Stage 8: Running benchmarks (%d runs each)", bench_runs)
        from src.benchmarking import run_benchmarks
        benchmark_summary = run_benchmarks(
            raw_dir=raw_dir,
            runs=bench_runs,
            max_workers=max_workers,
            spark_cfg={
                "app_name": cfg["spark"]["app_name"],
                "master": cfg["spark"]["master"],
                "driver_memory": cfg["spark"]["driver_memory"],
                "log_level": cfg["spark"]["log_level"],
            } if (args.spark or args.run_all) else {},
            output_dir=benchmarks_dir,
            include_gpu=(args.gpu or args.run_all),
        )
        # Regenerate benchmark charts
        from src.visualization import plot_benchmark
        plot_benchmark(benchmark_summary, out_dir=charts_dir)

    # ------------------------------------------------------------------ #
    # Stage 9: Report                                                      #
    # ------------------------------------------------------------------ #
    logger.info("Stage 9: Generating analytical summary report")
    from src.reporting import generate_analysis_summary
    report_path = generate_analysis_summary(
        cases=cases_all,
        testing=testing,
        vaccine=vaccine,
        state_comp=state_comp,
        monthly=monthly_all,
        anomalies=anomalies_all,
        benchmark_summary=benchmark_summary,
        focus_state=focus_state,
        output_path=os.path.join(out_reports, "analysis_summary.md"),
    )

    # ------------------------------------------------------------------ #
    # Done                                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Cleaned data : %s", processed_dir)
    logger.info("  Charts       : %s", charts_dir)
    logger.info("  Reports      : %s", out_reports)
    if benchmark_summary is not None:
        logger.info("  Benchmarks   : %s", benchmarks_dir)
    logger.info("  Dashboard    : streamlit run app.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
