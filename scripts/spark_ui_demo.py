"""
Spark UI Demo Script - Train a credit risk model and monitor in Spark UI.
This script demonstrates PySpark processing with visual monitoring.
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.utils.spark_config import create_spark_session, stop_spark_session


def run_spark_demo():
    """
    Run a complete Spark ML pipeline for demo purposes.
    This will show various stages in the Spark UI.
    """
    
    # Step 1: Create Spark session (UI will start)
    print("\n" + "="*80)
    print("STEP 1: Creating Spark Session")
    print("="*80)
    spark = create_spark_session(app_name="LoanRiskModel-Demo")
    
    input("\n⏸️  Press ENTER to continue to data loading...")
    
    # Step 2: Load data
    print("\n" + "="*80)
    print("STEP 2: Loading Sample Data")
    print("="*80)
    
    data_path = project_root / 'data' / 'sample' / 'loan_data_sample.csv'
    
    if not data_path.exists():
        print(f"❌ Sample data not found at {data_path}")
        print("   Run: python scripts/generate_sample_data.py first")
        spark.stop()
        return
    
    print(f"📂 Loading data from: {data_path}")
    df = spark.read.csv(str(data_path), header=True, inferSchema=True)
    
    # Cache the dataframe to see it in Storage tab
    df.cache()
    
    print(f"✅ Loaded {df.count()} records")
    print("\n📊 Schema:")
    df.printSchema()
    
    input("\n⏸️  Press ENTER to continue to data preprocessing...")
    
    # Step 3: Data Preprocessing
    print("\n" + "="*80)
    print("STEP 3: Data Preprocessing & Feature Engineering")
    print("="*80)
    
    # Select features for training
    feature_cols = [
        'credit_score', 'age', 'annual_income', 'loan_amount', 
        'loan_term', 'debt_to_income_ratio', 'num_credit_lines',
        'num_derogatory_marks', 'employment_length', 'avg_monthly_balance',
        'num_overdrafts', 'num_late_fees', 'monthly_income_deposits'
    ]
    
    # Handle missing values
    print("🔧 Handling missing values...")
    df_clean = df.na.fill({
        'months_since_last_delinquency': 999,
        'employment_length': 0
    })
    
    # Add derived features
    print("🔧 Creating derived features...")
    df_featured = df_clean.withColumn(
        'loan_to_income_ratio',
        F.col('loan_amount') / F.col('annual_income')
    ).withColumn(
        'monthly_payment_estimate',
        F.col('loan_amount') / F.col('loan_term')
    )
    
    feature_cols.extend(['loan_to_income_ratio', 'monthly_payment_estimate'])
    
    # Show sample
    print("\n📊 Sample records:")
    df_featured.select(feature_cols[:5] + ['loan_status']).show(5)
    
    input("\n⏸️  Press ENTER to continue to model training...")
    
    # Step 4: Build ML Pipeline
    print("\n" + "="*80)
    print("STEP 4: Building Machine Learning Pipeline")
    print("="*80)
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_raw"
    )
    
    # Scale features
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )
    
    # Train GBT Classifier
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="loan_status",
        maxDepth=6,
        maxIter=50,  # Reduced for faster demo
        stepSize=0.1,
        subsamplingRate=0.8,
        seed=42
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler, gbt])
    
    print("✅ Pipeline created with stages:")
    print("   1. VectorAssembler - Combine features")
    print("   2. StandardScaler - Normalize features")
    print("   3. GBTClassifier - Gradient Boosted Trees")
    
    input("\n⏸️  Press ENTER to start training (watch Spark UI)...")
    
    # Step 5: Train/Test Split
    print("\n" + "="*80)
    print("STEP 5: Splitting Data & Training Model")
    print("="*80)
    
    train_df, test_df = df_featured.randomSplit([0.8, 0.2], seed=42)
    train_df.cache()
    test_df.cache()
    
    print(f"📊 Training set: {train_df.count()} records")
    print(f"📊 Test set: {test_df.count()} records")
    
    print("\n🚀 Training model (this will generate multiple Spark jobs)...")
    print("   👀 Check Spark UI at http://localhost:4040 to see:")
    print("      - Jobs tab: See all Spark jobs")
    print("      - Stages tab: See task execution details")
    print("      - Storage tab: See cached RDDs/DataFrames")
    print("      - Environment tab: See all Spark configurations")
    print("      - Executors tab: See resource usage")
    
    start_time = time.time()
    model = pipeline.fit(train_df)
    training_time = time.time() - start_time
    
    print(f"\n✅ Model trained in {training_time:.2f} seconds")
    
    input("\n⏸️  Press ENTER to continue to evaluation...")
    
    # Step 6: Evaluate Model
    print("\n" + "="*80)
    print("STEP 6: Model Evaluation")
    print("="*80)
    
    print("🔮 Making predictions on test set...")
    predictions = model.transform(test_df)
    predictions.cache()
    
    # Show sample predictions
    print("\n📊 Sample predictions:")
    predictions.select(
        'credit_score', 'loan_amount', 'loan_status', 
        'prediction', 'probability'
    ).show(10, truncate=False)
    
    # Evaluate
    evaluator = BinaryClassificationEvaluator(
        labelCol="loan_status",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    auc_roc = evaluator.evaluate(predictions)
    
    evaluator.setMetricName("areaUnderPR")
    auc_pr = evaluator.evaluate(predictions)
    
    print("\n📈 Model Performance:")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   AUC-PR:  {auc_pr:.4f}")
    
    # Confusion matrix
    print("\n📊 Prediction Distribution:")
    predictions.groupBy('loan_status', 'prediction').count().show()
    
    input("\n⏸️  Press ENTER to continue to feature importance...")
    
    # Step 7: Feature Importance
    print("\n" + "="*80)
    print("STEP 7: Feature Importance Analysis")
    print("="*80)
    
    gbt_model = model.stages[-1]
    importance = gbt_model.featureImportances
    
    print("\n🎯 Top 10 Most Important Features:")
    feature_importance = list(zip(feature_cols, importance.toArray()))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, score) in enumerate(feature_importance[:10], 1):
        print(f"   {i:2d}. {feature:30s}: {score:.4f}")
    
    input("\n⏸️  Press ENTER to save model and finish...")
    
    # Step 8: Save Model
    print("\n" + "="*80)
    print("STEP 8: Saving Model")
    print("="*80)
    
    model_path = str(project_root / 'models' / 'spark_demo_model')
    print(f"💾 Saving model to: {model_path}")
    model.write().overwrite().save(model_path)
    print("✅ Model saved successfully")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📊 Summary:")
    print(f"   - Records processed: {df.count():,}")
    print(f"   - Features used: {len(feature_cols)}")
    print(f"   - Training time: {training_time:.2f} seconds")
    print(f"   - Model AUC-ROC: {auc_roc:.4f}")
    print(f"   - Model saved to: {model_path}")
    
    print("\n🌐 Spark UI Information:")
    print("   - Live UI: http://localhost:4040")
    print("   - Event logs saved for history server replay")
    
    input("\n⏸️  Press ENTER to stop Spark session and close UI...")
    
    # Cleanup
    print("\n🛑 Stopping Spark session...")
    spark.stop()
    print("✅ Spark session stopped")
    
    print("\n" + "="*80)
    print("Thank you for using the Spark UI Demo!")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                    🚀 SPARK UI DEMO - CREDIT RISK MODEL                    ║
║                                                                            ║
║  This demo will:                                                           ║
║  1. Start a Spark session (UI available at http://localhost:4040)         ║
║  2. Load sample loan application data                                     ║
║  3. Preprocess and engineer features                                      ║
║  4. Train a Gradient Boosted Trees model                                  ║
║  5. Evaluate model performance                                            ║
║  6. Save the trained model                                                ║
║                                                                            ║
║  📊 Watch the Spark UI to see:                                             ║
║     - Real-time job execution                                             ║
║     - Stage details and task metrics                                      ║
║     - SQL query plans and optimizations                                   ║
║     - Storage usage and cached data                                       ║
║     - Environment configurations                                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        run_spark_demo()
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
