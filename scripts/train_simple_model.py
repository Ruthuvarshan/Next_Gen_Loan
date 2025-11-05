"""
Simplified training script for XGBoost credit risk model.
Creates a complete pipeline with preprocessing and saves artifacts.

Usage:
    python scripts/train_simple_model.py --data data/processed/training_data.csv --target default
    python scripts/train_simple_model.py --data your_data.csv --target target_column --output models
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
import joblib


def infer_columns(df: pd.DataFrame, target: str):
    """Separate features from target and identify column types."""
    X = df.drop(columns=[target])
    y = df[target]
    
    # Identify categorical vs numeric columns
    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    
    return X, y, num_cols, cat_cols


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost credit risk model")
    ap.add_argument("--data", required=True, help="Path to CSV with training data")
    ap.add_argument("--target", default="target", help="Binary target column name (0=approve, 1=deny). Default: 'target'")
    ap.add_argument("--output", default="models", help="Output directory for model artifacts")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test set proportion (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = ap.parse_args()

    print("="*60)
    print("TRAINING CREDIT RISK MODEL")
    print("="*60)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f"\n1. Loading data from: {args.data}")
    df = pd.read_csv(args.data)
    print(f"   Dataset shape: {df.shape}")

    # Validate target column
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in dataset. Available columns: {list(df.columns)}")

    # Check if target is binary
    y_unique = df[args.target].unique()
    if len(y_unique) != 2:
        raise ValueError(f"Target must be binary (0/1). Found values: {sorted(y_unique)}")
    
    print(f"   Target column: '{args.target}'")
    print(f"   Class distribution: {dict(df[args.target].value_counts())}")

    # Separate features and target
    X, y, num_cols, cat_cols = infer_columns(df, args.target)
    
    print(f"\n2. Feature types identified:")
    print(f"   Numeric columns ({len(num_cols)}): {num_cols[:5]}{'...' if len(num_cols) > 5 else ''}")
    print(f"   Categorical columns ({len(cat_cols)}): {cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}")

    # Train-test split
    print(f"\n3. Splitting data (test size: {args.test_size})")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=args.seed, 
        stratify=y
    )
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")

    # Calculate class imbalance weight
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    if pos == 0 or neg == 0:
        raise ValueError("Train split has only one class. Provide more data or adjust split.")
    
    scale_pos_weight = float(neg) / float(pos)
    print(f"   Class imbalance ratio: {scale_pos_weight:.2f} (applying scale_pos_weight)")

    # Build preprocessing pipeline
    print(f"\n4. Building preprocessing pipeline")
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

    # Build XGBoost model
    print(f"\n5. Training XGBoost model")
    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        random_state=args.seed,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )

    # Complete pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    print("   Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    print("   ✓ Training complete")

    # Evaluate
    print(f"\n6. Evaluating on test set")
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n   ROC AUC: {auc:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, digits=4, target_names=["Approve", "Deny"]))
    print("\n   Confusion Matrix:")
    print(f"   TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"   FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

    # Save artifacts
    print(f"\n7. Saving model artifacts to: {args.output}")
    
    # Save joblib pipeline
    pipeline_path = os.path.join(args.output, "xgboost_model.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"   ✓ Saved: {pipeline_path}")
    
    # Save XGBoost model as JSON (for portability)
    try:
        model_step = pipeline.named_steps["model"]
        xgb_json_path = os.path.join(args.output, "xgb_model.json")
        model_step.get_booster().save_model(xgb_json_path)
        print(f"   ✓ Saved: {xgb_json_path}")
    except Exception as e:
        print(f"   Note: Could not export XGBoost JSON ({e})")

    # Save metadata
    meta_path = os.path.join(args.output, "model_metadata.json")
    metadata = {
        "created_at": int(time.time()),
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": os.path.abspath(args.data),
        "target_column": args.target,
        "test_size": args.test_size,
        "random_seed": args.seed,
        "metrics": {
            "roc_auc": float(auc),
            "confusion_matrix": cm.tolist()
        },
        "features": {
            "numeric": num_cols,
            "categorical": cat_cols,
            "total": len(num_cols) + len(cat_cols)
        },
        "model_config": {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.05,
            "scale_pos_weight": float(scale_pos_weight)
        },
        "artifacts": {
            "pipeline": os.path.abspath(pipeline_path),
            "xgb_json": os.path.abspath(xgb_json_path) if os.path.exists(os.path.join(args.output, "xgb_model.json")) else None
        }
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved: {meta_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {pipeline_path}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nTo use this model in the API:")
    print("1. Ensure models/xgboost_model.pkl exists")
    print("2. Start API: uvicorn src.api.main:app --reload")
    print("3. Visit: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
