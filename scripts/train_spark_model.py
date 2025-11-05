"""
PySpark-based model training script for credit risk assessment.
Handles end-to-end training pipeline with distributed processing.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

from src.utils.spark_config import create_spark_session, stop_spark_session
from src.utils.spark_preprocessing import SparkDataPreprocessor, train_test_split
from src.modules.spark_nlp_features import SparkNLPFeatureEngine
from src.modules.spark_risk_model import train_risk_model_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/spark_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(spark: SparkSession, data_path: str) -> DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        spark: SparkSession
        data_path: Path to CSV file
    
    Returns:
        Loaded DataFrame
    """
    logger.info(f"Loading data from {data_path}...")
    
    df = spark.read.csv(
        data_path,
        header=True,
        inferSchema=True,
        nullValue="NA"
    )
    
    row_count = df.count()
    col_count = len(df.columns)
    logger.info(f"Loaded {row_count} rows and {col_count} columns")
    
    return df


def preprocess_data(df: DataFrame) -> DataFrame:
    """
    Preprocess raw data with Spark ML transformations.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info("Preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = SparkDataPreprocessor()
    
    # Clean column names
    df = preprocessor.clean_column_names(df)
    
    # Remove duplicates
    original_count = df.count()
    df = preprocessor.remove_duplicates(df)
    new_count = df.count()
    logger.info(f"Removed {original_count - new_count} duplicate rows")
    
    # Identify feature types
    feature_types = preprocessor.identify_feature_types(df, exclude_cols=['loan_status', 'application_id'])
    logger.info(f"Detected {len(feature_types['numeric'])} numeric and {len(feature_types['categorical'])} categorical features")
    
    # Handle missing values
    df = preprocessor.handle_missing_values(
        df,
        numeric_cols=feature_types['numeric'],
        categorical_cols=feature_types['categorical']
    )
    
    return df, feature_types


def engineer_nlp_features(df: DataFrame, text_col: str = "bank_statement_text") -> DataFrame:
    """
    Extract NLP features from bank statement text.
    
    Args:
        df: DataFrame with bank statement text
        text_col: Name of text column
    
    Returns:
        DataFrame with NLP features added
    """
    if text_col not in df.columns:
        logger.warning(f"Column {text_col} not found. Skipping NLP feature engineering.")
        return df
    
    logger.info("Engineering NLP features from bank statements...")
    
    # Initialize NLP engine
    nlp_engine = SparkNLPFeatureEngine(df.sql_ctx.sparkSession)
    
    # Parse bank statements
    parsed_df = nlp_engine.parse_bank_statement(df, text_col)
    
    # Aggregate transaction features
    nlp_features_df = nlp_engine.aggregate_transaction_features(parsed_df, "application_id")
    
    # Join back to original DataFrame
    df = df.join(nlp_features_df, on="application_id", how="left")
    
    # Fill NLP feature nulls with 0
    nlp_cols = nlp_features_df.columns
    nlp_cols.remove("application_id")
    
    for col in nlp_cols:
        df = df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
    
    logger.info(f"Added {len(nlp_cols)} NLP features")
    
    return df


def create_preprocessing_pipeline(df: DataFrame, 
                                  feature_types: dict,
                                  target_col: str = "loan_status") -> tuple:
    """
    Create and fit preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        feature_types: Dictionary of feature types
        target_col: Target column name
    
    Returns:
        Tuple of (preprocessed_df, pipeline_model)
    """
    logger.info("Creating preprocessing pipeline...")
    
    preprocessor = SparkDataPreprocessor()
    
    # Create pipeline
    pipeline_model = preprocessor.create_preprocessing_pipeline(
        df,
        numeric_features=feature_types['numeric'],
        categorical_features=feature_types['categorical'],
        target_col=target_col
    )
    
    # Transform data
    df = pipeline_model.transform(df)
    
    return df, pipeline_model


def split_data(df: DataFrame, 
               test_size: float = 0.2,
               random_seed: int = 42) -> tuple:
    """
    Split data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Splitting data (test_size={test_size})...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify_col="loan_status"
    )
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    logger.info(f"Train set: {train_count} rows")
    logger.info(f"Test set: {test_count} rows")
    
    return train_df, test_df


def train_model(train_df: DataFrame,
               test_df: DataFrame,
               feature_cols: list,
               model_save_path: str) -> dict:
    """
    Train XGBoost risk model.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature columns
        model_save_path: Path to save model
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Training XGBoost model...")
    
    model, metrics = train_risk_model_pipeline(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        label_col="loan_status",
        model_save_path=model_save_path
    )
    
    logger.info("Training complete!")
    logger.info(f"Model metrics: {metrics}")
    
    return metrics


def main(args):
    """
    Main training pipeline.
    
    Args:
        args: Command line arguments
    """
    start_time = datetime.now()
    logger.info("="*80)
    logger.info("Starting PySpark Credit Risk Model Training")
    logger.info("="*80)
    
    # Create Spark session
    spark = create_spark_session(app_name="CreditRiskTraining")
    
    try:
        # 1. Load data
        df = load_data(spark, args.data_path)
        
        # 2. Preprocess data
        df, feature_types = preprocess_data(df)
        
        # 3. Engineer NLP features (if bank statement column exists)
        if args.with_nlp and "bank_statement_text" in df.columns:
            df = engineer_nlp_features(df, "bank_statement_text")
        
        # 4. Create preprocessing pipeline
        df, pipeline_model = create_preprocessing_pipeline(df, feature_types)
        
        # Save preprocessing pipeline
        pipeline_path = os.path.join(args.output_dir, "preprocessing_pipeline")
        pipeline_model.write().overwrite().save(pipeline_path)
        logger.info(f"Preprocessing pipeline saved to {pipeline_path}")
        
        # 5. Split data
        train_df, test_df = split_data(df, test_size=args.test_size, random_seed=args.seed)
        
        # 6. Get feature columns (all except target and ID columns)
        feature_cols = [
            col for col in df.columns 
            if col not in ['loan_status', 'application_id', 'features', 'rawPrediction', 
                          'probability', 'prediction']
        ]
        
        logger.info(f"Using {len(feature_cols)} features for training")
        
        # 7. Train model
        model_path = os.path.join(args.output_dir, "xgboost_model")
        metrics = train_model(train_df, test_df, feature_cols, model_path)
        
        # 8. Save metadata
        metadata = {
            "training_date": datetime.now().isoformat(),
            "data_path": args.data_path,
            "num_features": len(feature_cols),
            "train_rows": train_df.count(),
            "test_rows": test_df.count(),
            "metrics": metrics,
            "feature_columns": feature_cols
        }
        
        import json
        metadata_path = os.path.join(args.output_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model metadata saved to {metadata_path}")
        
        # Summary
        elapsed_time = (datetime.now() - start_time).total_seconds()
        logger.info("="*80)
        logger.info("Training Complete!")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Stop Spark session
        stop_spark_session()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PySpark Credit Risk Model")
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/loan_data.csv",
        help="Path to training data CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/",
        help="Directory to save trained model and artifacts"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--with-nlp",
        action="store_true",
        help="Enable NLP feature extraction from bank statements"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    main(args)
