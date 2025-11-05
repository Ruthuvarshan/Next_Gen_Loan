"""
PySpark-based NLP feature engineering for bank statement analysis.
Uses Spark ML and UDFs for distributed text processing.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SparkNLPFeatureEngine:
    """
    PySpark-based NLP feature extraction from bank statements.
    Analyzes transaction patterns and behavioral signals at scale.
    """
    
    # Transaction categories with keywords
    TRANSACTION_PATTERNS = {
        "salary": r"\b(salary|payroll|wages|income|pay\s*stub|employer|direct\s*deposit)\b",
        "debt_emi": r"\b(emi|loan\s*payment|credit\s*card|mortgage|installment|repayment)\b",
        "utility": r"\b(electric|water|gas|internet|phone|mobile|utility|bill)\b",
        "rent": r"\b(rent|lease|housing|landlord)\b",
        "overdraft": r"\b(overdraft|nsf|insufficient\s*funds|returned\s*check)\b",
        "late_fee": r"\b(late\s*fee|penalty|overdue|delinquent)\b",
        "payday_loan": r"\b(payday|cash\s*advance|short\s*term\s*loan)\b",
        "gambling": r"\b(casino|lottery|betting|poker|gambling)\b",
        "discretionary": r"\b(shopping|entertainment|restaurant|vacation|luxury)\b",
    }
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize Spark NLP engine.
        
        Args:
            spark: SparkSession instance
        """
        from src.utils.spark_config import get_or_create_spark
        self.spark = spark or get_or_create_spark()
        
        # Register UDFs
        self._register_udfs()
    
    def _register_udfs(self):
        """Register user-defined functions for transaction categorization."""
        
        def categorize_transaction(text: str) -> str:
            """Categorize a single transaction line."""
            if not text:
                return "unknown"
            
            text_lower = text.lower()
            
            # Check each category
            for category, pattern in self.TRANSACTION_PATTERNS.items():
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return category
            
            return "other"
        
        def extract_amount(text: str) -> float:
            """Extract monetary amount from text."""
            if not text:
                return 0.0
            
            # Find currency amounts ($1,234.56 or 1234.56)
            pattern = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
            matches = re.findall(pattern, text)
            
            if matches:
                # Remove $ and commas, convert to float
                amount_str = matches[0].replace('$', '').replace(',', '')
                try:
                    return float(amount_str)
                except ValueError:
                    return 0.0
            
            return 0.0
        
        def is_debit(text: str) -> int:
            """Check if transaction is a debit (expense)."""
            if not text:
                return 0
            
            debit_keywords = ['debit', 'withdrawal', 'payment', 'purchase', 'fee', 'charge']
            text_lower = text.lower()
            
            return 1 if any(kw in text_lower for kw in debit_keywords) else 0
        
        # Register UDFs
        self.categorize_udf = F.udf(categorize_transaction, StringType())
        self.extract_amount_udf = F.udf(extract_amount, DoubleType())
        self.is_debit_udf = F.udf(is_debit, IntegerType())
    
    def parse_bank_statement(self, df: DataFrame, text_col: str = "statement_text") -> DataFrame:
        """
        Parse bank statement text into structured transactions.
        
        Args:
            df: DataFrame with bank statement text column
            text_col: Name of the text column
        
        Returns:
            DataFrame with parsed transactions
        """
        # Split text into lines (transactions)
        df = df.withColumn("transaction_lines", F.split(F.col(text_col), r"\n"))
        
        # Explode to one row per transaction
        df = df.withColumn("transaction_text", F.explode(F.col("transaction_lines")))
        
        # Remove empty lines
        df = df.filter(F.length(F.col("transaction_text")) > 10)
        
        # Categorize each transaction
        df = df.withColumn("category", self.categorize_udf(F.col("transaction_text")))
        
        # Extract amount
        df = df.withColumn("amount", self.extract_amount_udf(F.col("transaction_text")))
        
        # Determine if debit
        df = df.withColumn("is_debit", self.is_debit_udf(F.col("transaction_text")))
        
        return df
    
    def aggregate_transaction_features(self, df: DataFrame, 
                                      group_col: str = "application_id") -> DataFrame:
        """
        Aggregate transaction-level data into application-level features.
        
        Args:
            df: DataFrame with parsed transactions
            group_col: Column to group by (application ID)
        
        Returns:
            DataFrame with aggregated features
        """
        # Group by application and calculate features
        agg_df = df.groupBy(group_col).agg(
            # Salary/Income features
            F.sum(F.when(F.col("category") == "salary", F.col("amount")).otherwise(0)).alias("total_salary_deposits"),
            F.count(F.when(F.col("category") == "salary", 1)).alias("salary_deposit_count"),
            F.avg(F.when(F.col("category") == "salary", F.col("amount"))).alias("avg_salary_deposit"),
            F.stddev(F.when(F.col("category") == "salary", F.col("amount"))).alias("salary_deposit_stddev"),
            
            # Debt/EMI features
            F.sum(F.when(F.col("category") == "debt_emi", F.col("amount")).otherwise(0)).alias("monthly_emi_total"),
            F.count(F.when(F.col("category") == "debt_emi", 1)).alias("emi_count"),
            
            # Utility features
            F.sum(F.when(F.col("category") == "utility", F.col("amount")).otherwise(0)).alias("monthly_utility_total"),
            
            # Rent features
            F.sum(F.when(F.col("category") == "rent", F.col("amount")).otherwise(0)).alias("monthly_rent"),
            
            # Risk flags
            F.count(F.when(F.col("category") == "overdraft", 1)).alias("overdraft_count"),
            F.count(F.when(F.col("category") == "late_fee", 1)).alias("late_fee_count"),
            F.count(F.when(F.col("category") == "payday_loan", 1)).alias("payday_loan_count"),
            F.count(F.when(F.col("category") == "gambling", 1)).alias("gambling_count"),
            
            # Discretionary spending
            F.sum(F.when(F.col("category") == "discretionary", F.col("amount")).otherwise(0)).alias("discretionary_spending"),
            
            # Overall patterns
            F.sum(F.col("amount")).alias("total_transaction_amount"),
            F.count("*").alias("transaction_count"),
            F.avg(F.col("amount")).alias("avg_transaction_amount"),
        )
        
        # Calculate derived features
        agg_df = agg_df.withColumn(
            "income_stability_variance",
            F.when(F.col("salary_deposit_stddev").isNotNull(), F.col("salary_deposit_stddev"))
            .otherwise(0.0)
        )
        
        agg_df = agg_df.withColumn(
            "risk_flag_count",
            F.col("overdraft_count") + F.col("late_fee_count") + 
            F.col("payday_loan_count") + F.col("gambling_count")
        )
        
        agg_df = agg_df.withColumn(
            "expense_to_income_ratio",
            F.when(F.col("total_salary_deposits") > 0,
                   (F.col("monthly_emi_total") + F.col("monthly_utility_total") + 
                    F.col("monthly_rent")) / F.col("total_salary_deposits"))
            .otherwise(0.0)
        )
        
        agg_df = agg_df.withColumn(
            "discretionary_ratio",
            F.when(F.col("total_salary_deposits") > 0,
                   F.col("discretionary_spending") / F.col("total_salary_deposits"))
            .otherwise(0.0)
        )
        
        # Fill nulls with 0
        numeric_cols = [col for col, dtype in agg_df.dtypes if dtype in ['double', 'int', 'bigint']]
        for col in numeric_cols:
            agg_df = agg_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)))
        
        return agg_df
    
    def extract_tfidf_features(self, df: DataFrame, 
                               text_col: str = "loan_purpose",
                               max_features: int = 100) -> DataFrame:
        """
        Extract TF-IDF features from loan purpose text.
        
        Args:
            df: Input DataFrame
            text_col: Text column name
            max_features: Maximum number of TF-IDF features
        
        Returns:
            DataFrame with TF-IDF features
        """
        # Tokenize
        tokenizer = RegexTokenizer(
            inputCol=text_col,
            outputCol="words",
            pattern="\\W"
        )
        df = tokenizer.transform(df)
        
        # Remove stop words
        remover = StopWordsRemover(
            inputCol="words",
            outputCol="filtered_words"
        )
        df = remover.transform(df)
        
        # Count vectorizer
        cv = CountVectorizer(
            inputCol="filtered_words",
            outputCol="raw_features",
            vocabSize=max_features
        )
        cv_model = cv.fit(df)
        df = cv_model.transform(df)
        
        # TF-IDF
        idf = IDF(
            inputCol="raw_features",
            outputCol="tfidf_features"
        )
        idf_model = idf.fit(df)
        df = idf_model.transform(df)
        
        return df
    
    def create_nlp_feature_pipeline(self, 
                                   text_col: str = "bank_statement_text",
                                   group_col: str = "application_id") -> DataFrame:
        """
        End-to-end NLP feature extraction pipeline.
        
        Args:
            text_col: Bank statement text column
            group_col: Application ID column
        
        Returns:
            DataFrame with NLP features
        """
        # This is a placeholder for a complete pipeline
        # In practice, you'd chain: parse -> aggregate -> tfidf
        pass


def extract_nlp_features_from_text(spark: SparkSession,
                                   text: str,
                                   application_id: str = "APP-001") -> Dict[str, float]:
    """
    Extract NLP features from a single bank statement text.
    Converts text to DataFrame, processes, and returns features as dict.
    
    Args:
        spark: SparkSession
        text: Bank statement text
        application_id: Application ID
    
    Returns:
        Dictionary of NLP features
    """
    # Create DataFrame
    df = spark.createDataFrame([(application_id, text)], ["application_id", "statement_text"])
    
    # Initialize engine
    engine = SparkNLPFeatureEngine(spark)
    
    # Parse transactions
    parsed_df = engine.parse_bank_statement(df, "statement_text")
    
    # Aggregate features
    feature_df = engine.aggregate_transaction_features(parsed_df, "application_id")
    
    # Convert to dictionary
    features = feature_df.first().asDict()
    
    # Remove application_id
    features.pop("application_id", None)
    
    return features
