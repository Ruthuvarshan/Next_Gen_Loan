"""
PySpark-based data preprocessing utilities.
Replaces pandas-based preprocessing with distributed Spark operations.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, Imputer, MinMaxScaler
)
from pyspark.ml import Pipeline
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SparkDataPreprocessor:
    """
    PySpark-based data preprocessing for loan applications.
    Handles missing values, scaling, encoding, and feature assembly.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the Spark preprocessor.
        
        Args:
            spark: SparkSession instance. If None, uses active session.
        """
        from src.utils.spark_config import get_or_create_spark
        self.spark = spark or get_or_create_spark()
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.pipeline: Optional[Pipeline] = None
    
    def identify_feature_types(self, df: DataFrame, target_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
        """
        Automatically identify numeric and categorical columns.
        
        Args:
            df: Input Spark DataFrame
            target_col: Target column to exclude from features
        
        Returns:
            Tuple of (numeric_columns, categorical_columns)
        """
        numeric_types = [IntegerType, LongType, FloatType, DoubleType, DecimalType]
        string_types = [StringType]
        
        numeric_cols = []
        categorical_cols = []
        
        for field in df.schema.fields:
            col_name = field.name
            
            # Skip target column
            if target_col and col_name == target_col:
                continue
            
            # Check data type
            dtype = type(field.dataType)
            
            if dtype in numeric_types:
                numeric_cols.append(col_name)
            elif dtype in string_types:
                categorical_cols.append(col_name)
        
        self.numeric_features = numeric_cols
        self.categorical_features = categorical_cols
        
        logger.info(f"Identified {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
        
        return numeric_cols, categorical_cols
    
    def handle_missing_values(self, df: DataFrame, 
                             numeric_strategy: str = "mean",
                             categorical_strategy: str = "mode") -> DataFrame:
        """
        Handle missing values using Spark ML Imputer.
        
        Args:
            df: Input DataFrame
            numeric_strategy: Strategy for numeric columns (mean, median, mode)
            categorical_strategy: Strategy for categorical columns (mode)
        
        Returns:
            DataFrame with imputed values
        """
        # Impute numeric columns
        if self.numeric_features:
            numeric_imputer = Imputer(
                strategy=numeric_strategy,
                inputCols=self.numeric_features,
                outputCols=[f"{col}_imputed" for col in self.numeric_features]
            )
            df = numeric_imputer.fit(df).transform(df)
            
            # Replace original columns
            for col in self.numeric_features:
                df = df.withColumn(col, F.col(f"{col}_imputed")).drop(f"{col}_imputed")
        
        # Impute categorical columns (fill with most frequent)
        if self.categorical_features:
            for col in self.categorical_features:
                # Get most frequent value
                mode_val = df.groupBy(col).count().orderBy(F.desc("count")).first()
                if mode_val:
                    mode_value = mode_val[0]
                    df = df.withColumn(col, F.when(F.col(col).isNull(), mode_value).otherwise(F.col(col)))
        
        return df
    
    def create_preprocessing_pipeline(self, 
                                     numeric_cols: List[str],
                                     categorical_cols: List[str],
                                     scale_numeric: bool = True,
                                     encode_categorical: bool = True) -> Pipeline:
        """
        Create a Spark ML Pipeline for preprocessing.
        
        Args:
            numeric_cols: List of numeric column names
            categorical_cols: List of categorical column names
            scale_numeric: Whether to scale numeric features
            encode_categorical: Whether to one-hot encode categorical features
        
        Returns:
            Spark ML Pipeline
        """
        stages = []
        
        # Impute missing values
        if numeric_cols:
            numeric_imputer = Imputer(
                strategy="mean",
                inputCols=numeric_cols,
                outputCols=[f"{col}_imputed" for col in numeric_cols]
            )
            stages.append(numeric_imputer)
            numeric_cols_imputed = [f"{col}_imputed" for col in numeric_cols]
        else:
            numeric_cols_imputed = []
        
        # Scale numeric features
        if scale_numeric and numeric_cols:
            assembler_numeric = VectorAssembler(
                inputCols=numeric_cols_imputed,
                outputCol="numeric_features_vector"
            )
            stages.append(assembler_numeric)
            
            scaler = StandardScaler(
                inputCol="numeric_features_vector",
                outputCol="scaled_numeric_features",
                withStd=True,
                withMean=True
            )
            stages.append(scaler)
        
        # Encode categorical features
        indexed_cols = []
        encoded_cols = []
        
        if encode_categorical and categorical_cols:
            for col in categorical_cols:
                # String indexing
                indexer = StringIndexer(
                    inputCol=col,
                    outputCol=f"{col}_indexed",
                    handleInvalid="keep"
                )
                stages.append(indexer)
                indexed_cols.append(f"{col}_indexed")
                
                # One-hot encoding
                encoder = OneHotEncoder(
                    inputCols=[f"{col}_indexed"],
                    outputCols=[f"{col}_encoded"],
                    dropLast=True
                )
                stages.append(encoder)
                encoded_cols.append(f"{col}_encoded")
        
        # Final vector assembler
        final_input_cols = []
        
        if scale_numeric and numeric_cols:
            final_input_cols.append("scaled_numeric_features")
        elif numeric_cols:
            final_input_cols.extend(numeric_cols_imputed)
        
        if encode_categorical and categorical_cols:
            final_input_cols.extend(encoded_cols)
        
        if final_input_cols:
            final_assembler = VectorAssembler(
                inputCols=final_input_cols,
                outputCol="features",
                handleInvalid="keep"
            )
            stages.append(final_assembler)
        
        pipeline = Pipeline(stages=stages)
        self.pipeline = pipeline
        
        return pipeline
    
    def normalize_column(self, df: DataFrame, col_name: str, 
                        new_col_name: Optional[str] = None) -> DataFrame:
        """
        Normalize a single column using Min-Max scaling.
        
        Args:
            df: Input DataFrame
            col_name: Column to normalize
            new_col_name: Name for normalized column (defaults to col_name)
        
        Returns:
            DataFrame with normalized column
        """
        if new_col_name is None:
            new_col_name = col_name
        
        # Assemble into vector
        assembler = VectorAssembler(inputCols=[col_name], outputCol=f"{col_name}_vec")
        df = assembler.transform(df)
        
        # Scale
        scaler = MinMaxScaler(inputCol=f"{col_name}_vec", outputCol=f"{col_name}_scaled")
        scaler_model = scaler.fit(df)
        df = scaler_model.transform(df)
        
        # Extract back to column
        df = df.withColumn(new_col_name, F.col(f"{col_name}_scaled")[0])
        
        # Drop temporary columns
        df = df.drop(f"{col_name}_vec", f"{col_name}_scaled")
        
        return df
    
    def clean_column_names(self, df: DataFrame) -> DataFrame:
        """
        Clean column names (remove special characters, spaces).
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with cleaned column names
        """
        for col in df.columns:
            new_col = col.strip().replace(" ", "_").replace("-", "_").replace(".", "_")
            if new_col != col:
                df = df.withColumnRenamed(col, new_col)
        
        return df
    
    def cast_numeric_columns(self, df: DataFrame, columns: List[str]) -> DataFrame:
        """
        Cast specified columns to DoubleType.
        
        Args:
            df: Input DataFrame
            columns: Columns to cast
        
        Returns:
            DataFrame with casted columns
        """
        for col in columns:
            if col in df.columns:
                df = df.withColumn(col, F.col(col).cast(DoubleType()))
        
        return df
    
    def remove_duplicates(self, df: DataFrame, subset: Optional[List[str]] = None) -> DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Subset of columns to consider for duplicates
        
        Returns:
            DataFrame without duplicates
        """
        if subset:
            return df.dropDuplicates(subset=subset)
        return df.dropDuplicates()
    
    def filter_outliers(self, df: DataFrame, col_name: str, 
                       method: str = "iqr", threshold: float = 1.5) -> DataFrame:
        """
        Filter outliers using IQR method.
        
        Args:
            df: Input DataFrame
            col_name: Column to filter
            method: Method to use (iqr, zscore)
            threshold: Threshold multiplier for IQR
        
        Returns:
            DataFrame without outliers
        """
        if method == "iqr":
            # Calculate quartiles
            quantiles = df.approxQuantile(col_name, [0.25, 0.75], 0.01)
            q1, q3 = quantiles[0], quantiles[1]
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Filter
            df = df.filter(
                (F.col(col_name) >= lower_bound) & 
                (F.col(col_name) <= upper_bound)
            )
        
        return df
    
    def train_test_split(self, df: DataFrame, 
                        test_size: float = 0.2,
                        seed: int = 42) -> Tuple[DataFrame, DataFrame]:
        """
        Split DataFrame into train and test sets.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            seed: Random seed
        
        Returns:
            Tuple of (train_df, test_df)
        """
        train_df, test_df = df.randomSplit([1 - test_size, test_size], seed=seed)
        
        return train_df, test_df


def load_csv_to_spark(file_path: str, spark: Optional[SparkSession] = None) -> DataFrame:
    """
    Load CSV file into Spark DataFrame with automatic schema inference.
    
    Args:
        file_path: Path to CSV file
        spark: SparkSession instance
    
    Returns:
        Spark DataFrame
    """
    from src.utils.spark_config import get_or_create_spark
    
    if spark is None:
        spark = get_or_create_spark()
    
    df = spark.read.csv(
        file_path,
        header=True,
        inferSchema=True
    )
    
    return df


def save_spark_dataframe(df: DataFrame, path: str, 
                        format: str = "parquet",
                        mode: str = "overwrite"):
    """
    Save Spark DataFrame to disk.
    
    Args:
        df: Spark DataFrame
        path: Output path
        format: Output format (parquet, csv, json)
        mode: Write mode (overwrite, append, error, ignore)
    """
    df.write.mode(mode).format(format).save(path)
