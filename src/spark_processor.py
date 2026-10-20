"""
spark_processor.py
------------------
Laboratory Concept 1: Apache Spark (PySpark)
=============================================

What is Apache Spark?
---------------------
Apache Spark is an open-source, distributed computing framework designed
for large-scale data processing.  It uses an in-memory, DAG-based
execution model that can be orders of magnitude faster than traditional
MapReduce for iterative algorithms.

Key Spark concepts used here
-----------------------------
* SparkSession   — entry point for Spark SQL and DataFrame APIs.
* Spark DataFrame — distributed, lazy, schema-aware collection.
* Transformations — filter(), groupBy(), agg(), withColumn(),
                    orderBy(), window functions — all lazy.
* Actions         — show(), collect(), write.csv() — trigger computation.
* Spark SQL       — SQL queries over registered temp views.
* Window functions — used to compute running sums and first differences.

Why Spark for public health data?
----------------------------------
Real-world outbreak monitoring involves streaming data from thousands of
testing centres, hospitals, and vaccination sites simultaneously. Spark's
distributed architecture can ingest and process millions of records per
second across a cluster, enabling near-real-time outbreak intelligence.
Even on a single laptop (local mode), Spark demonstrates the programming
model that scales linearly to a multi-node cluster by only changing the
'master' configuration string.

Local mode limitations
----------------------
This implementation runs in Spark local[*] mode (using all available
CPU cores on one machine).  It does NOT simulate a real cluster.
Spark's JVM startup and serialisation overhead means it will likely be
SLOWER than pandas on this small dataset.  That overhead is expected,
documented, and explained in the benchmarking section.  The value of
this implementation is the programming pattern, not raw speed on a 18k-
row file.
"""

import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _get_spark(app_name: str = "PublicHealthOutbreakMonitoring",
               master: str = "local[*]",
               driver_memory: str = "2g",
               log_level: str = "WARN"):
    """
    Create or retrieve an existing SparkSession.

    Configures the session for local execution on a student laptop.
    Handles Windows hostnames that contain underscores (which are not
    valid in Spark's internal URL scheme) by overriding SPARK_LOCAL_IP.
    """
    import os
    # Windows fix: hostnames with underscores break Spark's URL parser.
    # Force Spark to bind to 127.0.0.1 (loopback) instead of the hostname.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PySpark is not installed. Run: pip install pyspark==3.5.0"
        ) from e

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "4")   # small dataset — fewer shuffles
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(log_level)
    logger.info("SparkSession created: %s (master=%s)", app_name, master)
    return spark


def run_spark(
    raw_dir: str,
    app_name: str = "PublicHealthOutbreakMonitoring",
    master: str = "local[*]",
    driver_memory: str = "2g",
    log_level: str = "WARN",
    rolling_window: int = 7,
) -> Dict[str, Any]:
    """
    Execute the analytical pipeline using PySpark DataFrames and Spark SQL.

    Tasks performed
    ---------------
    1. Load the cleaned cases CSV into a Spark DataFrame.
    2. Select and rename relevant columns.
    3. Filter to Indian states (exclude unassigned rows).
    4. Compute total confirmed cases per state (grouped aggregation).
    5. Compute monthly new-case totals per state.
    6. Identify the single peak day per state (window function).
    7. Run a Spark SQL query for state comparison.
    8. Compute 7-day rolling average using Spark window functions.
    9. Return results as pandas DataFrames for downstream use.

    Parameters
    ----------
    raw_dir : str
        Directory containing the raw CSV files (used to locate the
        cleaned output; we re-clean inside Spark for demonstration).
    app_name, master, driver_memory, log_level : str
        SparkSession configuration.
    rolling_window : int
        Window size for rolling average computation.

    Returns
    -------
    dict
        Keys: 'state_totals', 'monthly_totals', 'peak_days',
              'rolling_avg', 'elapsed_seconds'
    """
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    t0 = time.perf_counter()
    spark = _get_spark(app_name, master, driver_memory, log_level)

    # ------------------------------------------------------------------
    # 1. Load the cleaned cases CSV (processed by pandas cleaning step)
    # ------------------------------------------------------------------
    cases_path = os.path.join(raw_dir, "..", "processed", "cleaned_cases.csv")
    cases_path = os.path.normpath(cases_path)

    if not os.path.exists(cases_path):
        # Fallback: load raw and do minimal Spark-side cleaning
        cases_path = os.path.join(raw_dir, "covid_19_india.csv")
        logger.warning(
            "cleaned_cases.csv not found — loading raw file: %s", cases_path
        )

    spark_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(cases_path.replace("\\", "/"))
    )

    # ------------------------------------------------------------------
    # 2. Select and cast columns
    # ------------------------------------------------------------------
    # Determine which columns are present (raw vs cleaned)
    cols = spark_df.columns
    state_col = "State" if "State" in cols else "State/UnionTerritory"
    has_daily = "DailyNewCases" in cols

    spark_df = spark_df.withColumnRenamed(state_col, "State")
    spark_df = spark_df.withColumn(
        "Date", F.to_date(F.col("Date"), "yyyy-MM-dd")
    )

    # Cast numeric columns
    for c in ["Confirmed", "Deaths", "Cured"]:
        if c in spark_df.columns:
            spark_df = spark_df.withColumn(c, F.col(c).cast("long"))

    if has_daily:
        for c in ["DailyNewCases", "DailyNewDeaths", "DailyNewRecoveries"]:
            if c in spark_df.columns:
                spark_df = spark_df.withColumn(c, F.col(c).cast("double"))

    # ------------------------------------------------------------------
    # 3. Filter: remove null States and dates; exclude unassigned tokens
    # ------------------------------------------------------------------
    exclude_states = [
        "Cases being reassigned to states",
        "Unassigned", "Miscellaneous", "State Unassigned"
    ]
    spark_df = spark_df.filter(
        F.col("State").isNotNull() &
        F.col("Date").isNotNull() &
        ~F.col("State").isin(exclude_states)
    )

    # Register as a Spark SQL temp view
    spark_df.createOrReplaceTempView("covid_cases")

    # ------------------------------------------------------------------
    # 4. State totals (Spark SQL — demonstrates SQL integration)
    # ------------------------------------------------------------------
    state_totals_spark = spark.sql("""
        SELECT
            State,
            MAX(Confirmed)  AS TotalConfirmed,
            MAX(Deaths)     AS TotalDeaths,
            MAX(Cured)      AS TotalRecoveries,
            MIN(Date)       AS DataStart,
            MAX(Date)       AS DataEnd,
            COUNT(*)        AS RecordCount
        FROM covid_cases
        GROUP BY State
        ORDER BY TotalConfirmed DESC
    """)

    # ------------------------------------------------------------------
    # 5. Monthly new-case totals per state using Spark DataFrame API
    # ------------------------------------------------------------------
    daily_col = "DailyNewCases" if has_daily else "Confirmed"

    monthly_spark = (
        spark_df
        .withColumn("YearMonth", F.date_format(F.col("Date"), "yyyy-MM"))
        .groupBy("State", "YearMonth")
        .agg(F.sum(daily_col).alias("MonthlyCases"))
        .orderBy("State", "YearMonth")
    )

    # ------------------------------------------------------------------
    # 6. Peak day per state (Window function)
    # ------------------------------------------------------------------
    if has_daily:
        window_rank = (
            Window.partitionBy("State")
            .orderBy(F.col("DailyNewCases").desc())
        )
        peak_df = (
            spark_df
            .withColumn("rank", F.rank().over(window_rank))
            .filter(F.col("rank") == 1)
            .select("State", "Date", "DailyNewCases")
            .withColumnRenamed("Date", "PeakDate")
            .withColumnRenamed("DailyNewCases", "PeakDailyNewCases")
        )
    else:
        peak_df = state_totals_spark.select("State")

    # ------------------------------------------------------------------
    # 7. 7-day rolling average using Spark Window (rows between)
    # ------------------------------------------------------------------
    if has_daily:
        window_rolling = (
            Window.partitionBy("State")
            .orderBy(F.col("Date").cast("long"))
            .rowsBetween(-(rolling_window - 1), 0)
        )
        rolling_spark = (
            spark_df
            .withColumn(
                "RollingAvg7d",
                F.avg(F.col("DailyNewCases")).over(window_rolling)
            )
            .select("State", "Date", "DailyNewCases", "RollingAvg7d")
            .orderBy("State", "Date")
        )
    else:
        rolling_spark = spark_df.select("State", "Date", "Confirmed")

    # ------------------------------------------------------------------
    # 8. Collect results to pandas for downstream compatibility
    # ------------------------------------------------------------------
    state_totals_pd = state_totals_spark.toPandas()
    monthly_pd = monthly_spark.toPandas()
    peak_pd = peak_df.toPandas()
    rolling_pd = rolling_spark.toPandas()

    t1 = time.perf_counter()
    elapsed = t1 - t0

    logger.info("Spark pipeline completed in %.3f s", elapsed)

    return {
        "state_totals": state_totals_pd,
        "monthly_totals": monthly_pd,
        "peak_days": peak_pd,
        "rolling_avg": rolling_pd,
        "elapsed_seconds": elapsed,
    }


def stop_spark():
    """Stop the active SparkSession if one exists."""
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        if spark is not None:
            spark.stop()
            logger.info("SparkSession stopped.")
    except Exception as e:
        logger.warning("Could not stop SparkSession: %s", e)
