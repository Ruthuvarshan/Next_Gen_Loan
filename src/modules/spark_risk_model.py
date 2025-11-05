"""
PySpark-based credit risk model using XGBoost4J-Spark for distributed training.
Handles large-scale training with Spark DataFrames.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SparkXGBoostRiskModel:
    """
    Distributed XGBoost model for credit risk assessment using PySpark.
    """
    
    def __init__(self, 
                 spark: Optional[SparkSession] = None,
                 feature_cols: Optional[List[str]] = None,
                 label_col: str = "loan_status"):
        """
        Initialize Spark XGBoost model.
        
        Args:
            spark: SparkSession instance
            feature_cols: List of feature column names
            label_col: Target label column name
        """
        from src.utils.spark_config import get_or_create_spark
        self.spark = spark or get_or_create_spark()
        self.feature_cols = feature_cols or []
        self.label_col = label_col
        self.model = None
        self.feature_importances = {}
    
    def prepare_features(self, df: DataFrame) -> DataFrame:
        """
        Prepare features by assembling into vector column.
        
        Args:
            df: Input DataFrame with separate feature columns
        
        Returns:
            DataFrame with 'features' vector column
        """
        # Auto-detect feature columns if not provided
        if not self.feature_cols:
            # All numeric columns except label
            self.feature_cols = [
                col for col, dtype in df.dtypes 
                if dtype in ['int', 'bigint', 'double', 'float'] and col != self.label_col
            ]
            logger.info(f"Auto-detected {len(self.feature_cols)} feature columns")
        
        # Assemble features
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features",
            handleInvalid="skip"  # Skip rows with invalid values
        )
        
        df = assembler.transform(df)
        
        return df
    
    def train_xgboost(self, 
                     train_df: DataFrame,
                     params: Optional[Dict] = None) -> PipelineModel:
        """
        Train XGBoost model using Spark ML.
        
        Note: For true XGBoost4J-Spark integration, you would use:
        from sparkxgb import XGBoostClassifier
        
        For now, we'll use Spark ML's GBTClassifier as a placeholder.
        
        Args:
            train_df: Training DataFrame with 'features' and label columns
            params: XGBoost parameters
        
        Returns:
            Trained pipeline model
        """
        from pyspark.ml.classification import GBTClassifier
        
        # Default parameters
        default_params = {
            "maxDepth": 6,
            "maxIter": 100,
            "stepSize": 0.1,
            "subsamplingRate": 0.8,
            "featureSubsetStrategy": "sqrt",
            "minInstancesPerNode": 10,
            "maxBins": 32
        }
        
        if params:
            default_params.update(params)
        
        # Create GBT classifier (Spark ML's gradient boosting)
        gbt = GBTClassifier(
            featuresCol="features",
            labelCol=self.label_col,
            predictionCol="prediction",
            probabilityCol="probability",
            rawPredictionCol="rawPrediction",
            **default_params
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[gbt])
        
        # Train model
        logger.info("Training XGBoost model with PySpark...")
        self.model = pipeline.fit(train_df)
        
        # Extract feature importances
        gbt_model = self.model.stages[0]
        importances = gbt_model.featureImportances.toArray()
        
        self.feature_importances = {
            feature: float(importance) 
            for feature, importance in zip(self.feature_cols, importances)
        }
        
        logger.info(f"Training complete. Top 5 features: {self._get_top_features(5)}")
        
        return self.model
    
    def _get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]
    
    def predict(self, df: DataFrame) -> DataFrame:
        """
        Make predictions on new data.
        
        Args:
            df: DataFrame with features
        
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_xgboost() first.")
        
        # Ensure features are assembled
        if "features" not in df.columns:
            df = self.prepare_features(df)
        
        # Make predictions
        predictions = self.model.transform(df)
        
        # Extract probability of positive class (default)
        predictions = predictions.withColumn(
            "default_probability",
            F.col("probability").getItem(1)
        )
        
        return predictions
    
    def evaluate(self, test_df: DataFrame) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            test_df: Test DataFrame
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        predictions = self.predict(test_df)
        
        # Binary classification metrics
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol=self.label_col,
            rawPredictionCol="rawPrediction"
        )
        
        auc_roc = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderROC"})
        auc_pr = binary_evaluator.evaluate(predictions, {binary_evaluator.metricName: "areaUnderPR"})
        
        # Multiclass metrics for accuracy, precision, recall
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol=self.label_col,
            predictionCol="prediction"
        )
        
        accuracy = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "accuracy"})
        precision = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedPrecision"})
        recall = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "weightedRecall"})
        f1 = multi_evaluator.evaluate(predictions, {multi_evaluator.metricName: "f1"})
        
        metrics = {
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def save_model(self, path: str):
        """
        Save trained model to disk.
        
        Args:
            path: Directory path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        self.model.write().overwrite().save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load trained model from disk.
        
        Args:
            path: Directory path to load model from
        """
        self.model = PipelineModel.load(path)
        logger.info(f"Model loaded from {path}")
        
        # Restore feature importances if available
        if self.model.stages:
            gbt_model = self.model.stages[0]
            if hasattr(gbt_model, 'featureImportances'):
                importances = gbt_model.featureImportances.toArray()
                self.feature_importances = {
                    feature: float(importance) 
                    for feature, importance in zip(self.feature_cols, importances)
                }
    
    def get_feature_importance_df(self) -> DataFrame:
        """
        Get feature importances as a Spark DataFrame.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if not self.feature_importances:
            raise ValueError("No feature importances available. Train model first.")
        
        # Convert to list of tuples
        importance_data = [
            (feature, importance) 
            for feature, importance in self.feature_importances.items()
        ]
        
        # Create DataFrame
        importance_df = self.spark.createDataFrame(
            importance_data,
            ["feature", "importance"]
        )
        
        # Sort by importance descending
        importance_df = importance_df.orderBy(F.desc("importance"))
        
        return importance_df


class SparkRiskScorer:
    """
    Production risk scorer using trained Spark XGBoost model.
    """
    
    def __init__(self, model_path: str, feature_cols: List[str]):
        """
        Initialize risk scorer with pre-trained model.
        
        Args:
            model_path: Path to saved model
            feature_cols: List of expected feature columns
        """
        from src.utils.spark_config import get_or_create_spark
        self.spark = get_or_create_spark()
        self.feature_cols = feature_cols
        
        # Load model
        self.model_wrapper = SparkXGBoostRiskModel(
            spark=self.spark,
            feature_cols=feature_cols
        )
        self.model_wrapper.load_model(model_path)
    
    def score_application(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Score a single loan application.
        
        Args:
            features: Dictionary of feature values
        
        Returns:
            Dictionary with prediction and probability
        """
        # Create DataFrame from features
        df = self.spark.createDataFrame([features])
        
        # Ensure all feature columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                df = df.withColumn(col, F.lit(0.0))
        
        # Make prediction
        predictions = self.model_wrapper.predict(df)
        
        # Extract result
        result = predictions.select("prediction", "default_probability").first()
        
        return {
            "prediction": int(result["prediction"]),
            "default_probability": float(result["default_probability"]),
            "approval_recommendation": "APPROVE" if result["prediction"] == 0 else "REJECT"
        }
    
    def batch_score(self, df: DataFrame) -> DataFrame:
        """
        Score multiple applications in batch.
        
        Args:
            df: DataFrame with application features
        
        Returns:
            DataFrame with predictions
        """
        return self.model_wrapper.predict(df)


def train_risk_model_pipeline(
    train_df: DataFrame,
    test_df: DataFrame,
    feature_cols: List[str],
    label_col: str = "loan_status",
    model_save_path: str = "models/spark_xgboost_model"
) -> Tuple[SparkXGBoostRiskModel, Dict[str, float]]:
    """
    Complete training pipeline for risk model.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        feature_cols: Feature column names
        label_col: Label column name
        model_save_path: Path to save trained model
    
    Returns:
        Tuple of (trained model, evaluation metrics)
    """
    # Initialize model
    model = SparkXGBoostRiskModel(
        feature_cols=feature_cols,
        label_col=label_col
    )
    
    # Prepare features
    logger.info("Preparing training features...")
    train_df = model.prepare_features(train_df)
    test_df = model.prepare_features(test_df)
    
    # Train model
    logger.info("Training XGBoost model...")
    model.train_xgboost(train_df)
    
    # Evaluate
    logger.info("Evaluating model...")
    metrics = model.evaluate(test_df)
    
    # Save model
    logger.info(f"Saving model to {model_save_path}...")
    model.save_model(model_save_path)
    
    return model, metrics
