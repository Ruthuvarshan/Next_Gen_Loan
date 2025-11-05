# PySpark Quick Reference

## Installation

```bash
pip install -r requirements.txt
```

## Training a Model

### Basic Training
```bash
python scripts/train_spark_model.py \
  --data-path data/processed/loan_data.csv \
  --output-dir models/
```

### With NLP Features
```bash
python scripts/train_spark_model.py \
  --data-path data/processed/loan_data.csv \
  --output-dir models/ \
  --with-nlp
```

### Custom Parameters
```bash
python scripts/train_spark_model.py \
  --data-path data/processed/loan_data.csv \
  --output-dir models/ \
  --test-size 0.3 \
  --seed 123 \
  --with-nlp
```

## Making Predictions

### Single Application Scoring
```python
from src.utils.spark_config import get_or_create_spark
from src.modules.spark_risk_model import SparkRiskScorer

# Initialize
spark = get_or_create_spark()
scorer = SparkRiskScorer(
    model_path="models/xgboost_model",
    feature_cols=['credit_score', 'age', 'loan_amount', 'loan_term', 'annual_income']
)

# Score
features = {
    'credit_score': 720,
    'age': 35,
    'loan_amount': 50000,
    'loan_term': 60,
    'annual_income': 75000
}

result = scorer.score_application(features)
print(result)
# Output: {'prediction': 0, 'default_probability': 0.23, 'approval_recommendation': 'APPROVE'}
```

### Batch Scoring
```python
from src.utils.spark_config import get_or_create_spark
from src.modules.spark_risk_model import SparkRiskScorer

# Initialize
spark = get_or_create_spark()
scorer = SparkRiskScorer(model_path="models/xgboost_model", feature_cols=[...])

# Load applications
applications_df = spark.read.csv("data/new_applications.csv", header=True, inferSchema=True)

# Score all at once
predictions = scorer.batch_score(applications_df)

# View results
predictions.select("application_id", "prediction", "default_probability").show(10)

# Save to CSV
predictions.write.csv("data/predictions.csv", header=True)
```

## Preprocessing

### SparkDataPreprocessor
```python
from src.utils.spark_config import get_or_create_spark
from src.utils.spark_preprocessing import SparkDataPreprocessor

spark = get_or_create_spark()
df = spark.read.csv("data/raw/data.csv", header=True, inferSchema=True)

preprocessor = SparkDataPreprocessor()

# Clean column names
df = preprocessor.clean_column_names(df)

# Remove duplicates
df = preprocessor.remove_duplicates(df)

# Identify feature types
feature_types = preprocessor.identify_feature_types(df, exclude_cols=['loan_status'])

# Handle missing values
df = preprocessor.handle_missing_values(
    df,
    numeric_cols=feature_types['numeric'],
    categorical_cols=feature_types['categorical']
)

# Create full preprocessing pipeline
pipeline_model = preprocessor.create_preprocessing_pipeline(
    df,
    numeric_features=feature_types['numeric'],
    categorical_features=feature_types['categorical'],
    target_col='loan_status'
)

# Transform data
df_transformed = pipeline_model.transform(df)
```

## NLP Features

### Extract Features from Bank Statement
```python
from src.utils.spark_config import get_or_create_spark
from src.modules.spark_nlp_features import extract_nlp_features_from_text

spark = get_or_create_spark()

bank_statement_text = """
Date: 2024-01-15
Employer Direct Deposit: $5,000.00
Date: 2024-01-16
Electric Company: -$120.50
Credit Card Payment: -$250.00
"""

features = extract_nlp_features_from_text(spark, bank_statement_text, "APP-001")
print(features)
```

### Parse Bank Statements at Scale
```python
from src.utils.spark_config import get_or_create_spark
from src.modules.spark_nlp_features import SparkNLPFeatureEngine

spark = get_or_create_spark()
engine = SparkNLPFeatureEngine(spark)

# Load applications with bank statements
df = spark.read.csv("data/applications_with_statements.csv", header=True)

# Parse transactions
parsed_df = engine.parse_bank_statement(df, "bank_statement_text")

# Aggregate to application-level features
feature_df = engine.aggregate_transaction_features(parsed_df, "application_id")

feature_df.show()
```

## Spark Session Management

### Create Session
```python
from src.utils.spark_config import create_spark_session

spark = create_spark_session(app_name="MyApp")
```

### Get or Create (Singleton)
```python
from src.utils.spark_config import get_or_create_spark

spark = get_or_create_spark()
```

### Stop Session
```python
from src.utils.spark_config import stop_spark_session

stop_spark_session()
```

## API Usage

### Start API
```bash
cd r:\SSF\Next_Gen_Loan
python src/api/main.py
```

### Check Model Type
```bash
# API automatically detects on startup:
# - If models/spark_xgboost_model exists → PySpark
# - Otherwise → Pandas

# Check logs:
# "PySpark models detected. Loading PySpark pipeline..."
# OR
# "Using pandas-based models..."
```

### Make Prediction (Same as Before)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer <jwt_token>" \
  -F "applicant_name=John Doe" \
  -F "credit_score=720" \
  -F "age=35" \
  -F "loan_amount=50000" \
  -F "loan_term=60" \
  -F "annual_income=75000"
```

## Distributed Deployment

### YARN (Hadoop Cluster)
```bash
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --executor-memory 8g \
  --executor-cores 4 \
  --num-executors 10 \
  --py-files src.zip \
  scripts/train_spark_model.py \
  --data-path hdfs:///data/loan_data.csv \
  --output-dir hdfs:///models/
```

### Kubernetes
```bash
spark-submit \
  --master k8s://https://kubernetes.default.svc:443 \
  --deploy-mode cluster \
  --executor-memory 8g \
  --executor-cores 4 \
  --num-executors 10 \
  --conf spark.kubernetes.container.image=spark:3.5.0-python3.11 \
  scripts/train_spark_model.py \
  --data-path s3a://bucket/loan_data.csv \
  --output-dir s3a://bucket/models/
```

### Local Multi-Core
```bash
# Uses all CPU cores by default (local[*])
python scripts/train_spark_model.py --data-path data/loan_data.csv
```

## Performance Tuning

### Increase Memory
```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("CreditRisk") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.memory", "16g") \
    .getOrCreate()
```

### Adjust Partitions
```python
# Increase shuffle partitions (2-3x number of cores)
spark.conf.set("spark.sql.shuffle.partitions", "400")

# Repartition DataFrame
df = df.repartition(200)
```

### Cache DataFrames
```python
# Cache frequently accessed data
df.cache()
train_df.cache()
test_df.cache()

# Unpersist when done
df.unpersist()
```

### Use Broadcast Joins
```python
from pyspark.sql.functions import broadcast

# Broadcast small lookup tables
df_large = df_large.join(broadcast(df_small), "key")
```

## Troubleshooting

### Out of Memory
```python
# Increase executor memory
spark.conf.set("spark.executor.memory", "16g")

# Or use more partitions (smaller batches)
df = df.repartition(400)
```

### Slow Performance
```python
# Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")

# Increase shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "400")

# Cache intermediate results
df.cache()
```

### Driver Out of Memory
```python
# Don't collect large DataFrames
# Instead of: df.collect()
# Use: df.write.csv("output.csv")

# Or sample data
df.sample(0.1).collect()
```

### Connection Timeout
```python
# Increase network timeout
spark.conf.set("spark.network.timeout", "600s")
spark.conf.set("spark.sql.broadcastTimeout", "600")
```

## Common Operations

### Read/Write Data
```python
# Read CSV
df = spark.read.csv("data.csv", header=True, inferSchema=True)

# Read Parquet (more efficient)
df = spark.read.parquet("data.parquet")

# Write CSV
df.write.csv("output.csv", header=True, mode="overwrite")

# Write Parquet
df.write.parquet("output.parquet", mode="overwrite")
```

### DataFrame Operations
```python
# Select columns
df.select("col1", "col2").show()

# Filter rows
df.filter(df.age > 30).show()

# Group and aggregate
df.groupBy("category").agg({"value": "mean"}).show()

# Join DataFrames
df1.join(df2, df1.key == df2.key, "inner")

# Sort
df.orderBy("column", ascending=False)
```

### View Data
```python
# Show first 20 rows
df.show()

# Show first 10 rows
df.show(10)

# Show full columns (no truncation)
df.show(truncate=False)

# Print schema
df.printSchema()

# Count rows
df.count()

# Get column names
df.columns

# Describe statistics
df.describe().show()
```

## Monitoring

### Spark UI
Access at: `http://localhost:4040`

Shows:
- Job progress
- Stage details
- Storage usage
- Executor metrics
- SQL query plans

### Logging
```python
import logging

# Set log level
spark.sparkContext.setLogLevel("WARN")  # Options: ALL, DEBUG, INFO, WARN, ERROR, FATAL, OFF

# Or use Python logging
logger = logging.getLogger(__name__)
logger.info("Processing started")
```

## Resources

- **PySpark Docs**: https://spark.apache.org/docs/latest/api/python/
- **Spark ML Guide**: https://spark.apache.org/docs/latest/ml-guide.html
- **Migration Guide**: See `PYSPARK_MIGRATION.md`
- **Complete Summary**: See `PYSPARK_MIGRATION_COMPLETE.md`
