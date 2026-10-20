"""
benchmarking.py
---------------
Performance benchmarking for sequential, multithreaded, and Spark
implementations of the same analytical task.

Measurement approach
---------------------
- Each implementation is run RUNS times (default 3).
- Wall-clock time is measured with time.perf_counter().
- Median execution time is used as the representative figure to
  reduce the influence of outliers (e.g. JVM warm-up on first Spark run).
- Memory usage is measured with tracemalloc where available.

Formulas
---------
    Speedup(parallel vs sequential) = T_sequential / T_parallel
    Efficiency(threads)             = Speedup / n_workers
    (Efficiency for Spark local[*] is not directly equivalent to
     Python thread efficiency; this is noted in the report.)

Important caveats
-----------------
- Spark overhead on a small dataset (~18 k rows) typically EXCEEDS its
  benefit, resulting in Speedup < 1.  This is expected and documented.
- Multithreading is limited by Python's GIL for CPU-bound operations.
- All timings are from a single laptop; they should not be extrapolated
  to multi-node cluster performance.
- Results are saved to CSV and PNG for the academic report.
"""

import logging
import os
import time
import tracemalloc
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-run timer helper
# ---------------------------------------------------------------------------

def _timed_run(fn, *args, **kwargs):
    """Run fn(*args, **kwargs), return (result, elapsed_seconds, peak_kb)."""
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, t1 - t0, peak / 1024  # peak in KB


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmarks(
    raw_dir: str,
    runs: int = 3,
    max_workers: int = 4,
    spark_cfg: Optional[Dict[str, str]] = None,
    output_dir: str = "outputs/benchmarks",
    include_gpu: bool = True,
) -> pd.DataFrame:
    """
    Benchmark sequential, multithreaded, Spark, and GPU implementations.

    Parameters
    ----------
    raw_dir : str
        Path to raw data directory.
    runs : int
        Number of repeated runs per implementation.
    max_workers : int
        Thread-pool size for the multithreaded run.
    spark_cfg : dict, optional
        Spark configuration overrides.
    output_dir : str
        Directory to save benchmark results CSV.

    Returns
    -------
    pd.DataFrame
        Benchmark summary with columns:
        Implementation, Run, ElapsedSeconds, PeakMemoryKB
    """
    from src.sequential_processor import run_sequential
    from src.threaded_processor import run_threaded

    spark_cfg = spark_cfg or {}
    records: List[Dict[str, Any]] = []

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Sequential                                                           #
    # ------------------------------------------------------------------ #
    logger.info("Benchmarking sequential (%d runs)...", runs)
    seq_times = []
    for i in range(runs):
        _, elapsed, peak_kb = _timed_run(run_sequential, raw_dir)
        records.append({
            "Implementation": "Sequential (pandas)",
            "Run": i + 1,
            "ElapsedSeconds": round(elapsed, 4),
            "PeakMemoryKB": round(peak_kb, 1),
        })
        seq_times.append(elapsed)
        logger.info("  Sequential run %d: %.3f s", i + 1, elapsed)

    seq_median = sorted(seq_times)[len(seq_times) // 2]

    # ------------------------------------------------------------------ #
    # Multithreaded                                                        #
    # ------------------------------------------------------------------ #
    logger.info("Benchmarking multithreaded (%d runs, %d workers)...", runs, max_workers)
    thr_times = []
    for i in range(runs):
        result, elapsed, peak_kb = _timed_run(
            run_threaded, raw_dir, max_workers=max_workers
        )
        # Use parallel section time for fair comparison
        par_elapsed = result.get("elapsed_seconds", elapsed)
        records.append({
            "Implementation": f"Multithreaded ({max_workers} workers)",
            "Run": i + 1,
            "ElapsedSeconds": round(par_elapsed, 4),
            "PeakMemoryKB": round(peak_kb, 1),
        })
        thr_times.append(par_elapsed)
        logger.info("  Threaded run %d: %.3f s (parallel section)", i + 1, par_elapsed)

    thr_median = sorted(thr_times)[len(thr_times) // 2]

    # ------------------------------------------------------------------ #
    # Spark                                                                #
    # ------------------------------------------------------------------ #
    logger.info("Benchmarking Spark (%d runs)...", runs)
    spark_times = []
    try:
        from src.spark_processor import run_spark, stop_spark
        for i in range(runs):
            _, elapsed, peak_kb = _timed_run(
                run_spark,
                raw_dir,
                **spark_cfg
            )
            records.append({
                "Implementation": "Spark (local[*])",
                "Run": i + 1,
                "ElapsedSeconds": round(elapsed, 4),
                "PeakMemoryKB": round(peak_kb, 1),
            })
            spark_times.append(elapsed)
            logger.info("  Spark run %d: %.3f s", i + 1, elapsed)
        stop_spark()
    except Exception as e:
        logger.error("Spark benchmark failed: %s", e)
        for i in range(runs):
            records.append({
                "Implementation": "Spark (local[*])",
                "Run": i + 1,
                "ElapsedSeconds": None,
                "PeakMemoryKB": None,
            })
        spark_times = [None] * runs

    spark_median = (
        sorted(t for t in spark_times if t is not None)[len([t for t in spark_times if t is not None]) // 2]
        if any(t is not None for t in spark_times)
        else None
    )

    # ------------------------------------------------------------------ #
    # GPU Hardware Acceleration (NVIDIA CUDA)                             #
    # ------------------------------------------------------------------ #
    gpu_times = []
    gpu_impl_name = "GPU (CUDA)"
    if include_gpu:
        try:
            from src.gpu_processor import is_gpu_available, run_gpu, get_gpu_info
            gpu_info = get_gpu_info()
            if gpu_info["available"]:
                gpu_impl_name = f"GPU ({gpu_info['name']})"
                logger.info("Benchmarking GPU [%s] (%d runs)...", gpu_info["name"], runs)
                for i in range(runs):
                    _, elapsed, peak_kb = _timed_run(run_gpu, raw_dir)
                    records.append({
                        "Implementation": gpu_impl_name,
                        "Run": i + 1,
                        "ElapsedSeconds": round(elapsed, 4),
                        "PeakMemoryKB": round(peak_kb, 1),
                    })
                    gpu_times.append(elapsed)
                    logger.info("  GPU run %d: %.3f s", i + 1, elapsed)
            else:
                logger.info("GPU hardware acceleration skipped (no CUDA device found).")
        except Exception as e:
            logger.warning("GPU benchmark execution failed: %s", e)

    # ------------------------------------------------------------------ #
    # Build results DataFrame                                              #
    # ------------------------------------------------------------------ #
    df_runs = pd.DataFrame(records)

    # Summary statistics
    summary_rows = []
    targets = [
        ("Sequential (pandas)", seq_times),
        (f"Multithreaded ({max_workers} workers)", thr_times),
        ("Spark (local[*])", [t for t in spark_times if t is not None]),
    ]
    if gpu_times:
        targets.append((gpu_impl_name, gpu_times))

    for impl, times in targets:
        if times:
            med = sorted(times)[len(times) // 2]
            speedup = round(seq_median / med, 3) if med and seq_median else None
        else:
            med = None
            speedup = None

        efficiency = None
        if speedup is not None and impl.startswith("Multithreaded"):
            efficiency = round(speedup / max_workers, 3)

        summary_rows.append({
            "Implementation": impl,
            "Runs": runs,
            "MedianElapsedSeconds": round(med, 4) if med else None,
            "SpeedupVsSequential": speedup,
            "Efficiency": efficiency,
        })

    df_summary = pd.DataFrame(summary_rows)

    # Save
    runs_path = os.path.join(output_dir, "benchmark_runs.csv")
    summary_path = os.path.join(output_dir, "benchmark_summary.csv")
    df_runs.to_csv(runs_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    logger.info("Benchmark results saved to %s", output_dir)

    logger.info("\n%s", df_summary.to_string(index=False))
    return df_summary
