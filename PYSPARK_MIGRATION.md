# PySpark Migration Guide

## Overview

This document outlines the complete migration of the Next-Gen Loan Origination System from **pandas/scikit-learn** to **PySpark** for distributed data processing and model training.

## Architecture Changes

### Before (Pandas-based)
```
Data → pandas DataFrame → sklearn preprocessing → XGBoost training → joblib model → FastAPI
```

### After (PySpark-based)
```
Data → Spark DataFrame → Spark ML Pipeline → GBTClassifier/XGBoost4J-Spark → PipelineModel → FastAPI
```

## Key Components

### 1. PySpark Dependencies

Added to `requirements.txt`:
```
pyspark==3.5.0
pyarrow==14.0.1
spark-nlp==5.2.0
```

### 2. Spark Configuration (`src/utils/spark_config.py`)

**Purpose**: Centralized Spark session management

**Key Functions**:
- `create_spark_session(app_name)`: Creates SparkSession with optimized configuration
- `get_or_create_spark()`: Singleton pattern for global session
- `stop_spark_session()`: Proper resource cleanup
- `read_csv_spark()`, `write_spark_dataframe()`: I/O utilities

**Configuration**:
- Master: `local[*]` (use all CPU cores)
- Executor memory: 4g
- Driver memory: 4g
- Arrow optimization: enabled
- Adaptive query execution: enabled
- Warehouse directory: `spark-warehouse/`

### 3. Spark Preprocessing (`src/utils/spark_preprocessing.py`)

**Purpose**: Replace pandas preprocessing with Spark ML pipelines

**SparkDataPreprocessor Class**:

#### Core Methods:
- `identify_feature_types(df, exclude_cols)`: Auto-detect numeric vs categorical columns
- `handle_missing_values(df, numeric_cols, categorical_cols)`: Spark ML Imputer with mean/mode strategies
- `create_preprocessing_pipeline(df, numeric_features, categorical_features, target_col)`: Full Spark ML pipeline

#### Pipeline Stages:
1. **Imputer**: Handle missing values (mean for numeric, mode for categorical)
2. **StringIndexer**: Convert categorical strings to numeric indices
3. **OneHotEncoder**: One-hot encode categorical features
4. **StandardScaler**: Normalize numeric features
5. **VectorAssembler**: Combine all features into single vector column

#### Utility Functions:
- `train_test_split(df, test_size, random_state, stratify_col)`: Stratified splitting for Spark DataFrames
- `normalize_column(df, col_name, method)`: Min-max or z-score normalization
- `clean_column_names(df)`: Remove special characters from column names
- `remove_duplicates(df, subset)`: Distributed deduplication
- `filter_outliers(df, col_name, n_std)`: IQR-based outlier removal

### 4. Spark NLP Features (`src/modules/spark_nlp_features.py`)

**Purpose**: Distributed NLP feature extraction from bank statements

**SparkNLPFeatureEngine Class**:

#### Transaction Categorization UDFs:
- `categorize_transaction(text)`: Regex-based transaction categorization
- `extract_amount(text)`: Extract monetary amounts from text
- `is_debit(text)`: Identify debit transactions

#### Transaction Categories:
- Salary/Income
- Debt/EMI payments
- Utility bills
- Rent
- Overdrafts (risk flag)
- Late fees (risk flag)
- Payday loans (risk flag)
- Gambling (risk flag)
- Discretionary spending

#### Core Methods:
- `parse_bank_statement(df, text_col)`: Parse text into structured transactions
- `aggregate_transaction_features(df, group_col)`: Aggregate to application-level features
- `extract_tfidf_features(df, text_col, max_features)`: TF-IDF on loan purpose text

#### Generated Features:
- `total_salary_deposits`, `salary_deposit_count`, `avg_salary_deposit`
- `monthly_emi_total`, `emi_count`
- `monthly_utility_total`, `monthly_rent`
- `overdraft_count`, `late_fee_count`, `payday_loan_count`, `gambling_count`
- `discretionary_spending`
- `income_stability_variance`
- `risk_flag_count`
- `expense_to_income_ratio`
- `discretionary_ratio`

### 5. Spark Risk Model (`src/modules/spark_risk_model.py`)

**Purpose**: Distributed XGBoost training with Spark ML

**SparkXGBoostRiskModel Class**:

#### Core Methods:
- `prepare_features(df)`: Assemble features into vector column
- `train_xgboost(train_df, params)`: Train GBTClassifier (or XGBoost4J-Spark)
- `predict(df)`: Make predictions on new data
- `evaluate(test_df)`: Calculate AUC-ROC, AUC-PR, accuracy, precision, recall, F1
- `save_model(path)`: Save PipelineModel to disk
- `load_model(path)`: Load PipelineModel from disk
- `get_feature_importance_df()`: Get feature importances as DataFrame

#### Model Parameters:
- `maxDepth`: 6
- `maxIter`: 100
- `stepSize`: 0.1
- `subsamplingRate`: 0.8
- `featureSubsetStrategy`: "sqrt"
- `minInstancesPerNode`: 10
- `maxBins`: 32

**SparkRiskScorer Class**:
- Production scorer for single application or batch scoring
- `score_application(features)`: Score single loan application
- `batch_score(df)`: Batch scoring for multiple applications

### 6. Spark Training Script (`scripts/train_spark_model.py`)

**Purpose**: End-to-end distributed training pipeline

#### Pipeline Steps:
1. **Load data**: Read CSV into Spark DataFrame
2. **Preprocess**: Clean column names, remove duplicates, handle missing values
3. **Engineer NLP features**: Parse bank statements, aggregate transaction features
4. **Create preprocessing pipeline**: Fit Spark ML pipeline
5. **Split data**: Stratified train/test split
6. **Train model**: Train GBTClassifier with Spark ML
7. **Evaluate**: Calculate metrics on test set
8. **Save artifacts**: Save model, pipeline, and metadata

#### Command Line Arguments:
```bash
python scripts/train_spark_model.py \
  --data-path data/processed/loan_data.csv \
  --output-dir models/ \
  --test-size 0.2 \
  --seed 42 \
  --with-nlp
```

#### Output Artifacts:
- `models/xgboost_model/`: Trained PipelineModel
- `models/preprocessing_pipeline/`: Preprocessing PipelineModel
- `models/model_metadata.json`: Training metadata and feature names

### 7. API Integration (`src/api/main.py`)

**Purpose**: Dual-mode API supporting both pandas and PySpark models

#### Startup Logic:
1. Check if PySpark models exist (`models/spark_xgboost_model`)
2. If yes: Load Spark models, set `state.use_pyspark = True`
3. If no: Load pandas models, set `state.use_pyspark = False`

#### PySpark Prediction Flow:
```python
if state.use_pyspark:
    # Extract NLP features with Spark
    nlp_features = extract_nlp_features_from_text(spark, bank_text, applicant_id)
    
    # Combine features
    all_features = {**traditional_features, **idp_features, **nlp_features}
    
    # Score with SparkRiskScorer
    result = state.spark_risk_scorer.score_application(all_features)
    
    prediction = result['prediction']
    probability = result['default_probability']
```

## Migration Checklist

### ✅ Completed
- [x] Add PySpark dependencies to `requirements.txt`
- [x] Create `src/utils/spark_config.py` - Spark session management
- [x] Create `src/utils/spark_preprocessing.py` - SparkDataPreprocessor with full Spark ML pipeline
- [x] Create `src/modules/spark_nlp_features.py` - SparkNLPFeatureEngine with UDFs and aggregation
- [x] Create `src/modules/spark_risk_model.py` - SparkXGBoostRiskModel and SparkRiskScorer
- [x] Create `scripts/train_spark_model.py` - End-to-end training pipeline
- [x] Update `src/api/main.py` - Dual-mode API with PySpark support

### ⏳ Testing & Validation
- [ ] Test Spark session creation and configuration
- [ ] Test SparkDataPreprocessor with sample data
- [ ] Test SparkNLPFeatureEngine transaction parsing
- [ ] Test SparkXGBoostRiskModel training and evaluation
- [ ] Run full training pipeline with sample dataset
- [ ] Test API prediction endpoint with PySpark model
- [ ] Load test distributed processing performance
- [ ] Validate feature importance extraction
- [ ] Test model save/load functionality
- [ ] Validate preprocessing pipeline persistence

### 📚 Documentation
- [ ] Update README.md with PySpark setup instructions
- [ ] Update QUICKSTART.md with PySpark training commands
- [ ] Create distributed deployment guide
- [ ] Document Spark cluster configuration
- [ ] Add performance tuning guide

## Usage Examples

### Training a Model

```bash
# Install dependencies
pip install -r requirements.txt

# Train with PySpark
python scripts/train_spark_model.py \
  --data-path data/processed/loan_data.csv \
  --output-dir models/ \
  --test-size 0.2 \
  --with-nlp

# Outputs:
# - models/xgboost_model/ (Spark ML PipelineModel)
# - models/preprocessing_pipeline/ (Spark ML Pipeline)
# - models/model_metadata.json (metadata and feature names)
```

### Making Predictions

```python
from src.utils.spark_config import get_or_create_spark
from src.modules.spark_risk_model import SparkRiskScorer

# Initialize
spark = get_or_create_spark()
scorer = SparkRiskScorer(
    model_path="models/xgboost_model",
    feature_cols=['credit_score', 'age', 'loan_amount', ...]
)

# Score application
features = {
    'credit_score': 720,
    'age': 35,
    'loan_amount': 50000,
    'loan_term': 60,
    'annual_income': 75000,
    # ... more features
}

result = scorer.score_application(features)
print(result)
# {'prediction': 0, 'default_probability': 0.23, 'approval_recommendation': 'APPROVE'}
```

### Batch Scoring

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BatchScoring").getOrCreate()

# Load applications
applications_df = spark.read.csv("data/new_applications.csv", header=True, inferSchema=True)

# Batch score
predictions = scorer.batch_score(applications_df)

# Show results
predictions.select("application_id", "prediction", "default_probability").show()
```

## Performance Optimization

### Spark Configuration

For production deployments, adjust Spark configuration in `src/utils/spark_config.py`:

```python
# Cluster mode
spark = SparkSession.builder \
    .appName("CreditRisk") \
    .master("spark://master:7077") \  # Use cluster master
    .config("spark.executor.memory", "16g") \  # Increase for larger datasets
    .config("spark.executor.cores", "4") \
    .config("spark.executor.instances", "10") \
    .config("spark.sql.shuffle.partitions", "400") \  # 2-3x num cores
    .getOrCreate()
```

### Data Partitioning

```python
# Repartition for better parallelism
df = df.repartition(200)  # Or use hash partitioning on key column
df = df.repartition("application_id")
```

### Caching

```python
# Cache frequently accessed DataFrames
df.cache()
train_df.cache()
test_df.cache()
```

## XGBoost4J-Spark Integration

To use native distributed XGBoost instead of GBTClassifier:

1. Install XGBoost4J-Spark:
```bash
# Add to requirements.txt
xgboost-spark==2.0.0
```

2. Replace GBTClassifier in `src/modules/spark_risk_model.py`:
```python
from sparkxgb import XGBoostClassifier

xgb = XGBoostClassifier(
    featuresCol="features",
    labelCol="loan_status",
    predictionCol="prediction",
    max_depth=6,
    eta=0.1,
    num_round=100,
    num_workers=4
)
```

## Distributed Deployment

### Running on YARN

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

### Running on Kubernetes

```bash
spark-submit \
  --master k8s://https://kubernetes.default.svc:443 \
  --deploy-mode cluster \
  --executor-memory 8g \
  --executor-cores 4 \
  --num-executors 10 \
  --conf spark.kubernetes.container.image=spark:3.5.0-python3.11 \
  scripts/train_spark_model.py
```

## Benefits of PySpark Migration

### Scalability
- **Horizontal scaling**: Process datasets larger than memory by distributing across cluster
- **Parallel processing**: Utilize all CPU cores (local) or cluster nodes (distributed)
- **Data partitioning**: Automatically partition data for efficient processing

### Performance
- **Lazy evaluation**: Optimize execution plans before running
- **Catalyst optimizer**: Query optimization for DataFrame operations
- **Tungsten engine**: Memory management and code generation

### Production Readiness
- **Fault tolerance**: Automatic task retry on failure
- **Monitoring**: Spark UI for job tracking and debugging
- **Integration**: Works with HDFS, S3, Kafka, Cassandra, etc.

## Troubleshooting

### Out of Memory Errors
```python
# Increase executor memory
spark.conf.set("spark.executor.memory", "16g")

# Or reduce batch size
df = df.repartition(400)  # More partitions = smaller batches
```

### Slow Shuffle Operations
```python
# Increase shuffle partitions
spark.conf.set("spark.sql.shuffle.partitions", "400")

# Enable adaptive query execution
spark.conf.set("spark.sql.adaptive.enabled", "true")
```

### Driver Out of Memory
```python
# Avoid collecting large DataFrames
# Instead of: df.collect()
# Use: df.write.csv("output.csv")

# Or sample data
df.sample(0.1).collect()
```

## References

- PySpark Documentation: https://spark.apache.org/docs/latest/api/python/
- Spark ML Guide: https://spark.apache.org/docs/latest/ml-guide.html
- XGBoost4J-Spark: https://xgboost.readthedocs.io/en/stable/jvm/xgboost4j_spark_tutorial.html
- Spark Configuration: https://spark.apache.org/docs/latest/configuration.html
