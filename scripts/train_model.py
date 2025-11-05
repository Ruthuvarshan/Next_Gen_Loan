"""
Training script for the credit risk model with fairness mitigation.

Usage:
    python scripts/train_model.py --data data/processed/training_data.csv --output models/
    python scripts/train_model.py --data data/processed/training_data.csv --fairness --constraint equalized_odds
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime

from src.modules.risk_model import CreditRiskModel
from src.modules.fairness_audit import FairnessAuditor
from src.modules.xai_explainer import XAIExplainer
from src.utils.config import settings
from src.utils.preprocessing import DataPreprocessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train credit risk model')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/', help='Output directory for models')
    parser.add_argument('--fairness', action='store_true', help='Enable fairness mitigation')
    parser.add_argument('--constraint', type=str, default='equalized_odds',
                       choices=['demographic_parity', 'equalized_odds'],
                       help='Fairness constraint type')
    parser.add_argument('--test-split', type=float, default=0.2, help='Test set proportion')
    parser.add_argument('--use-smote', action='store_true', default=True, help='Use SMOTE for imbalance')
    parser.add_argument('--hyperparameter-tuning', action='store_true', help='Enable hyperparameter tuning')
    
    return parser.parse_args()


def load_and_prepare_data(data_path: str, test_split: float = 0.2):
    """Load and prepare training data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features, target, and sensitive attributes
    target_col = 'default'  # Assuming this is the target column
    sensitive_cols = settings.sensitive_attributes_list
    
    # Features (everything except target and sensitive attributes)
    feature_cols = [col for col in df.columns if col != target_col and col not in sensitive_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    sensitive_attrs = df[sensitive_cols] if all(col in df.columns for col in sensitive_cols) else None
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    
    if sensitive_attrs is not None:
        X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
            X, y, sensitive_attrs, test_size=test_split, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        sens_train, sens_test = None, None
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Default rate in training: {y_train.mean():.2%}")
    print(f"Default rate in test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test, sens_train, sens_test


def train_baseline_model(X_train, y_train, X_test, y_test, use_smote=True, hyperparameter_tuning=False):
    """Train baseline model without fairness constraints."""
    print("\n" + "="*60)
    print("TRAINING BASELINE MODEL")
    print("="*60)
    
    model = CreditRiskModel()
    
    # Train
    training_metrics = model.train(
        X_train,
        y_train,
        use_smote=use_smote,
        hyperparameter_tuning=hyperparameter_tuning
    )
    
    # Evaluate
    print("\nEvaluating baseline model...")
    eval_metrics = model.evaluate(X_test, y_test)
    
    return model, training_metrics, eval_metrics


def audit_and_mitigate_fairness(
    baseline_model,
    X_train,
    y_train,
    X_test,
    y_test,
    sens_train,
    sens_test,
    constraint='equalized_odds'
):
    """Perform fairness audit and mitigation."""
    print("\n" + "="*60)
    print("FAIRNESS AUDIT AND MITIGATION")
    print("="*60)
    
    # Initialize auditor
    auditor = FairnessAuditor(sensitive_features=settings.sensitive_attributes_list)
    
    # Audit baseline model
    print("\n1. Auditing baseline model...")
    baseline_metrics, baseline_summary = auditor.audit_baseline_model(
        baseline_model.model,
        X_test,
        y_test,
        sens_test
    )
    
    # Check if mitigation is needed
    needs_mitigation = any(
        not metrics['is_fair_eo'] or not metrics['is_fair_dp']
        for metrics in baseline_summary.values()
    )
    
    if not needs_mitigation:
        print("\n✓ Baseline model meets fairness criteria. No mitigation needed.")
        return baseline_model, baseline_metrics, baseline_summary, None, None
    
    print("\n2. Applying fairness mitigation with GridSearch...")
    
    # Create base model for mitigation
    from xgboost import XGBClassifier
    base_model = XGBClassifier(
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        objective='binary:logistic',
        random_state=42
    )
    
    # Apply GridSearch mitigation
    mitigated_model = auditor.mitigate_with_grid_search(
        base_model,
        X_train,
        y_train,
        sens_train,
        constraint=constraint,
        grid_size=50
    )
    
    # Audit mitigated model
    print("\n3. Auditing mitigated model...")
    mitigated_metrics, mitigated_summary = auditor.audit_mitigated_model(
        mitigated_model,
        X_test,
        y_test,
        sens_test
    )
    
    # Generate comparison report
    print("\n4. Generating fairness comparison report...")
    fairness_report = auditor.generate_fairness_report(
        baseline_summary,
        mitigated_summary
    )
    
    return mitigated_model, mitigated_metrics, mitigated_summary, fairness_report, auditor


def generate_xai_visualizations(model, X_test, y_test, output_dir):
    """Generate and save XAI visualizations."""
    print("\n" + "="*60)
    print("GENERATING XAI VISUALIZATIONS")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize explainer
    print("Initializing SHAP explainer...")
    explainer = XAIExplainer(model.model if isinstance(model, CreditRiskModel) else model)
    
    # Sample background data
    background_sample = X_test.sample(min(100, len(X_test)), random_state=42)
    explainer.initialize_explainer(background_sample)
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    shap_values = explainer.calculate_shap_values(X_test)
    
    # Global summary plot
    print("Generating global SHAP summary plot...")
    explainer.plot_global_summary(
        X_test,
        plot_type='beeswarm',
        save_path=str(output_dir / 'shap_global_summary.png')
    )
    
    # Example waterfall plots for denied cases
    print("Generating example waterfall plots...")
    predictions = model.predict(X_test)[0]
    denied_indices = np.where(predictions == 1)[0][:3]  # First 3 denied cases
    
    for i, idx in enumerate(denied_indices):
        explainer.plot_waterfall(
            X_test,
            instance_index=idx,
            save_path=str(output_dir / f'shap_waterfall_denied_{i+1}.png')
        )
    
    print(f"XAI visualizations saved to {output_dir}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*60)
    print("CREDIT RISK MODEL TRAINING PIPELINE")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Fairness mitigation: {args.fairness}")
    print(f"Constraint: {args.constraint}")
    print("="*60)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, sens_train, sens_test = load_and_prepare_data(
        args.data,
        test_split=args.test_split
    )
    
    # Train baseline model
    baseline_model, training_metrics, eval_metrics = train_baseline_model(
        X_train,
        y_train,
        X_test,
        y_test,
        use_smote=args.use_smote,
        hyperparameter_tuning=args.hyperparameter_tuning
    )
    
    # Save baseline model
    baseline_path = output_dir / 'xgboost_baseline_model.pkl'
    baseline_model.save_model(baseline_path)
    print(f"\nBaseline model saved to {baseline_path}")
    
    # Fairness audit and mitigation
    final_model = baseline_model
    fairness_report = None
    
    if args.fairness and sens_train is not None and sens_test is not None:
        mitigated_model, mitigated_metrics, mitigated_summary, fairness_report, auditor = audit_and_mitigate_fairness(
            baseline_model,
            X_train,
            y_train,
            X_test,
            y_test,
            sens_train,
            sens_test,
            constraint=args.constraint
        )
        
        if mitigated_model is not None:
            # Save mitigated model
            final_model = mitigated_model
            mitigated_path = output_dir / 'xgboost_mitigated_model.pkl'
            joblib.dump(mitigated_model, mitigated_path)
            print(f"\nMitigated model saved to {mitigated_path}")
            
            # Save fairness report
            if fairness_report is not None:
                report_path = output_dir / 'fairness_report.csv'
                fairness_report.to_csv(report_path, index=False)
                print(f"Fairness report saved to {report_path}")
    
    # Generate XAI visualizations
    docs_dir = output_dir.parent / 'docs' / 'images'
    generate_xai_visualizations(final_model, X_test, y_test, docs_dir)
    
    # Save final model as the production model
    final_path = output_dir / 'xgboost_model.pkl'
    if isinstance(final_model, CreditRiskModel):
        final_model.save_model(final_path)
    else:
        joblib.dump(final_model, final_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final model saved to {final_path}")
    print(f"ROC AUC: {eval_metrics['roc_auc']:.4f}")
    print(f"F1 Score: {eval_metrics['f1_score']:.4f}")
    
    if fairness_report is not None:
        print("\nFairness improvements:")
        print(fairness_report.to_string(index=False))


if __name__ == "__main__":
    main()
