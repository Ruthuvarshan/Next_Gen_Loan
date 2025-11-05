"""
PySpark configuration and session management.
Creates and manages Spark sessions for distributed data processing.
"""

import os
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from typing import Optional

# Spark configuration constants
SPARK_APP_NAME = "NextGenLoanOriginationSystem"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")  # local[*] uses all cores
SPARK_EXECUTOR_MEMORY = os.getenv("SPARK_EXECUTOR_MEMORY", "4g")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "4g")
SPARK_WAREHOUSE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "spark-warehouse")

# Create warehouse directory if it doesn't exist
os.makedirs(SPARK_WAREHOUSE_DIR, exist_ok=True)


def create_spark_session(app_name: Optional[str] = None) -> SparkSession:
    """
    Create and configure a Spark session for the loan origination system.
    
    Args:
        app_name: Optional custom application name. Defaults to SPARK_APP_NAME.
    
    Returns:
        Configured SparkSession instance
    """
    if app_name is None:
        app_name = SPARK_APP_NAME
    
    # Create Spark configuration
    conf = SparkConf()
    conf.setAppName(app_name)
    conf.setMaster(SPARK_MASTER)
    conf.set("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
    conf.set("spark.driver.memory", SPARK_DRIVER_MEMORY)
    conf.set("spark.sql.warehouse.dir", SPARK_WAREHOUSE_DIR)
    
    # Arrow optimization for pandas conversion
    conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    
    # Adaptive Query Execution for better performance
    conf.set("spark.sql.adaptive.enabled", "true")
    conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
    
    # Shuffle partitions (adjust based on data size)
    conf.set("spark.sql.shuffle.partitions", "200")
    
    # Create or get existing session
    spark = SparkSession.builder.config(conf=conf).getOrCreate()
    
    # Set log level to WARN to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def get_spark_session() -> SparkSession:
    """
    Get or create the global Spark session.
    
    Returns:
        SparkSession instance
    """
    return SparkSession.builder.getOrCreate()


def stop_spark_session():
    """
    Stop the active Spark session and release resources.
    """
    spark = SparkSession.builder.getOrCreate()
    spark.stop()


# Global Spark session (lazy initialization)
_spark_session: Optional[SparkSession] = None


def get_or_create_spark() -> SparkSession:
    """
    Get the global Spark session or create it if it doesn't exist.
    
    Returns:
        SparkSession instance
    """
    global _spark_session
    
    if _spark_session is None or _spark_session._jsc.sc().isStopped():
        _spark_session = create_spark_session()
    
    return _spark_session


def read_csv_spark(file_path: str, header: bool = True, infer_schema: bool = True) -> "DataFrame":
    """
    Read CSV file into Spark DataFrame.
    
    Args:
        file_path: Path to CSV file
        header: Whether CSV has header row
        infer_schema: Whether to infer schema automatically
    
    Returns:
        Spark DataFrame
    """
    spark = get_or_create_spark()
    
    return spark.read.csv(
        file_path,
        header=header,
        inferSchema=infer_schema
    )


def write_spark_dataframe(df: "DataFrame", path: str, mode: str = "overwrite", format: str = "parquet"):
    """
    Write Spark DataFrame to disk.
    
    Args:
        df: Spark DataFrame to write
        path: Output path
        mode: Write mode (overwrite, append, error, ignore)
        format: Output format (parquet, csv, json, delta)
    """
    df.write.mode(mode).format(format).save(path)


if __name__ == "__main__":
    # Test Spark session creation
    print("Creating Spark session...")
    spark = create_spark_session()
    
    print(f"Spark version: {spark.version}")
    print(f"Spark master: {spark.sparkContext.master}")
    print(f"Spark app name: {spark.sparkContext.appName}")
    
    # Test simple DataFrame operation
    test_data = [(1, "test", 100), (2, "example", 200)]
    df = spark.createDataFrame(test_data, ["id", "name", "value"])
    
    print("\nTest DataFrame:")
    df.show()
    
    print("\nSpark session created successfully!")
    
    stop_spark_session()
    print("Spark session stopped.")
