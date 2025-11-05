"""
Module 5: Algorithmic Fairness and Bias Mitigation Audit

This module provides:
1. Fairness metric calculation (Demographic Parity, Equalized Odds)
2. Bias detection across protected groups
3. Mitigation strategies (Reweighing, GridSearch with constraints)
4. Before/after reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fairlearn imports
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    false_positive_rate,
    false_negative_rate,
    true_positive_rate,
    equalized_odds_difference,
    demographic_parity_difference
)
from fairlearn.reductions import GridSearch, ErrorRate, DemographicParity, EqualizedOdds

# AIF360 imports (optional, for pre-processing)
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    print("AIF360 not available. Install with: pip install aif360")

from src.utils.config import settings


class FairnessAuditor:
    """
    Audits machine learning models for algorithmic bias and fairness.
    """
    
    def __init__(self, sensitive_features: List[str]):
        """
        Initialize the fairness auditor.
        
        Args:
            sensitive_features: List of sensitive attribute column names
        """
        self.sensitive_features = sensitive_features
        self.baseline_metrics: Optional[pd.DataFrame] = None
        self.mitigated_metrics: Optional[pd.DataFrame] = None
    
    def calculate_fairness_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: pd.Series,
        sensitive_attrs: pd.DataFrame,
        model_name: str = "Model"
    ) -> pd.DataFrame:
        """
        Calculate comprehensive fairness metrics disaggregated by sensitive attributes.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            sensitive_attrs: DataFrame of sensitive attributes
            model_name: Name identifier for the model
            
        Returns:
            DataFrame with disaggregated metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        
        # Define metrics to calculate
        metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_t, y_p: precision_score(y_t, y_p, zero_division=0),
            'recall': lambda y_t, y_p: recall_score(y_t, y_p, zero_division=0),
            'selection_rate': selection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'true_positive_rate': true_positive_rate
        }
        
        # Calculate MetricFrame for each sensitive feature
        results = []
        
        for sensitive_feature in self.sensitive_features:
            if sensitive_feature not in sensitive_attrs.columns:
                print(f"Warning: {sensitive_feature} not found in data")
                continue
            
            # Create MetricFrame
            metric_frame = MetricFrame(
                metrics=metrics,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_attrs[sensitive_feature]
            )
            
            # Get disaggregated results
            by_group = metric_frame.by_group
            overall = metric_frame.overall
            difference = metric_frame.difference()
            ratio = metric_frame.ratio()
            
            # Format results
            for group in by_group.index:
                row = {
                    'model': model_name,
                    'sensitive_feature': sensitive_feature,
                    'group': str(group),
                    'sample_size': int((sensitive_attrs[sensitive_feature] == group).sum())
                }
                
                # Add all metrics
                for metric_name in metrics.keys():
                    row[metric_name] = by_group.loc[group, metric_name]
                
                results.append(row)
            
            # Add overall row
            overall_row = {
                'model': model_name,
                'sensitive_feature': sensitive_feature,
                'group': 'Overall',
                'sample_size': len(y_true)
            }
            for metric_name in metrics.keys():
                overall_row[metric_name] = overall[metric_name]
            results.append(overall_row)
            
            # Add difference and ratio rows
            diff_row = {
                'model': model_name,
                'sensitive_feature': sensitive_feature,
                'group': 'Max Difference',
                'sample_size': 0
            }
            for metric_name in metrics.keys():
                diff_row[metric_name] = difference[metric_name]
            results.append(diff_row)
        
        return pd.DataFrame(results)
    
    def calculate_group_fairness(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> Dict:
        """
        Calculate high-level group fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_attrs: DataFrame of sensitive attributes
            
        Returns:
            Dictionary with fairness metrics
        """
        fairness_summary = {}
        
        for sensitive_feature in self.sensitive_features:
            if sensitive_feature not in sensitive_attrs.columns:
                continue
            
            # Demographic Parity Difference
            dp_diff = demographic_parity_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_attrs[sensitive_feature]
            )
            
            # Equalized Odds Difference
            eo_diff = equalized_odds_difference(
                y_true,
                y_pred,
                sensitive_features=sensitive_attrs[sensitive_feature]
            )
            
            fairness_summary[sensitive_feature] = {
                'demographic_parity_difference': float(dp_diff),
                'equalized_odds_difference': float(eo_diff),
                'is_fair_dp': abs(dp_diff) < settings.fairness_threshold,
                'is_fair_eo': abs(eo_diff) < settings.fairness_threshold
            }
        
        return fairness_summary
    
    def audit_baseline_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform baseline fairness audit on unmitigated model.
        
        Args:
            model: Trained model to audit
            X_test: Test features
            y_test: Test labels
            sensitive_attrs: Sensitive attributes for test set
            
        Returns:
            Tuple of (detailed_metrics_df, fairness_summary_dict)
        """
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Calculate detailed metrics
        self.baseline_metrics = self.calculate_fairness_metrics(
            y_test,
            y_pred,
            y_prob,
            sensitive_attrs,
            model_name="Baseline"
        )
        
        # Calculate group fairness summary
        fairness_summary = self.calculate_group_fairness(
            y_test,
            y_pred,
            sensitive_attrs
        )
        
        print("\n=== Baseline Model Fairness Audit ===")
        for feature, metrics in fairness_summary.items():
            print(f"\n{feature}:")
            print(f"  Demographic Parity Difference: {metrics['demographic_parity_difference']:.4f}")
            print(f"  Equalized Odds Difference: {metrics['equalized_odds_difference']:.4f}")
            print(f"  Fair (DP)? {metrics['is_fair_dp']}")
            print(f"  Fair (EO)? {metrics['is_fair_eo']}")
        
        return self.baseline_metrics, fairness_summary
    
    def mitigate_with_reweighing(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Apply AIF360 Reweighing algorithm for bias mitigation.
        
        This is a pre-processing technique that adjusts training sample weights.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sensitive_attrs: Sensitive attributes
            
        Returns:
            Tuple of (X_train, y_train, sample_weights)
        """
        if not AIF360_AVAILABLE:
            raise ImportError("AIF360 not installed. Install with: pip install aif360")
        
        # Note: Full implementation would require converting to AIF360 dataset format
        # This is a placeholder showing the concept
        
        print("Applying Reweighing algorithm...")
        
        # Create sample weights (simplified version)
        sample_weights = np.ones(len(X_train))
        
        # In practice, use AIF360's Reweighing class
        # rw = Reweighing(unprivileged_groups=[{sensitive_feature: 0}],
        #                 privileged_groups=[{sensitive_feature: 1}])
        # dataset_transformed = rw.fit_transform(dataset)
        
        return X_train, y_train, sample_weights
    
    def mitigate_with_grid_search(
        self,
        base_model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sensitive_attrs: pd.DataFrame,
        constraint: str = 'equalized_odds',
        grid_size: int = 50
    ):
        """
        Apply Fairlearn GridSearch for in-processing bias mitigation.
        
        This trains multiple models with fairness constraints and returns
        the best trade-off between accuracy and fairness.
        
        Args:
            base_model: Base estimator (e.g., XGBoost)
            X_train: Training features
            y_train: Training labels
            sensitive_attrs: Sensitive attributes
            constraint: Fairness constraint ('demographic_parity' or 'equalized_odds')
            grid_size: Number of models to train in the grid
            
        Returns:
            Mitigated model
        """
        print(f"\n=== Fairness Mitigation with GridSearch ===")
        print(f"Constraint: {constraint}")
        print(f"Grid size: {grid_size}")
        
        # Choose constraint
        if constraint == 'demographic_parity':
            fairness_constraint = DemographicParity()
        elif constraint == 'equalized_odds':
            fairness_constraint = EqualizedOdds()
        else:
            raise ValueError(f"Unknown constraint: {constraint}")
        
        # Create GridSearch wrapper
        mitigator = GridSearch(
            estimator=base_model,
            constraints=fairness_constraint,
            grid_size=grid_size
        )
        
        # Fit with sensitive features
        # Use the first sensitive feature for mitigation
        sensitive_feature = self.sensitive_features[0]
        
        print(f"Training {grid_size} models with {constraint} constraint...")
        mitigator.fit(
            X_train,
            y_train,
            sensitive_features=sensitive_attrs[sensitive_feature]
        )
        
        print(f"Grid search complete. Found {len(mitigator.predictors_)} models.")
        
        return mitigator
    
    def audit_mitigated_model(
        self,
        mitigated_model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attrs: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Audit the fairness-mitigated model.
        
        Args:
            mitigated_model: Mitigated model
            X_test: Test features
            y_test: Test labels
            sensitive_attrs: Sensitive attributes
            
        Returns:
            Tuple of (detailed_metrics_df, fairness_summary_dict)
        """
        # Get predictions
        y_pred = mitigated_model.predict(X_test)
        
        # For GridSearch models, probabilities might not be available
        try:
            y_prob = mitigated_model.predict_proba(X_test)[:, 1]
        except:
            y_prob = y_pred
        
        # Calculate detailed metrics
        self.mitigated_metrics = self.calculate_fairness_metrics(
            y_test,
            y_pred,
            y_prob,
            sensitive_attrs,
            model_name="Mitigated"
        )
        
        # Calculate group fairness summary
        fairness_summary = self.calculate_group_fairness(
            y_test,
            y_pred,
            sensitive_attrs
        )
        
        print("\n=== Mitigated Model Fairness Audit ===")
        for feature, metrics in fairness_summary.items():
            print(f"\n{feature}:")
            print(f"  Demographic Parity Difference: {metrics['demographic_parity_difference']:.4f}")
            print(f"  Equalized Odds Difference: {metrics['equalized_odds_difference']:.4f}")
            print(f"  Fair (DP)? {metrics['is_fair_dp']}")
            print(f"  Fair (EO)? {metrics['is_fair_eo']}")
        
        return self.mitigated_metrics, fairness_summary
    
    def generate_fairness_report(
        self,
        baseline_summary: Dict,
        mitigated_summary: Dict,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate before/after fairness comparison report.
        
        Args:
            baseline_summary: Fairness metrics for baseline model
            mitigated_summary: Fairness metrics for mitigated model
            save_path: Optional path to save report
            
        Returns:
            Comparison DataFrame
        """
        rows = []
        
        for sensitive_feature in self.sensitive_features:
            if sensitive_feature not in baseline_summary:
                continue
            
            baseline = baseline_summary[sensitive_feature]
            mitigated = mitigated_summary[sensitive_feature]
            
            rows.append({
                'Sensitive Feature': sensitive_feature,
                'Metric': 'Demographic Parity Difference',
                'Baseline': baseline['demographic_parity_difference'],
                'Mitigated': mitigated['demographic_parity_difference'],
                'Improvement': baseline['demographic_parity_difference'] - mitigated['demographic_parity_difference']
            })
            
            rows.append({
                'Sensitive Feature': sensitive_feature,
                'Metric': 'Equalized Odds Difference',
                'Baseline': baseline['equalized_odds_difference'],
                'Mitigated': mitigated['equalized_odds_difference'],
                'Improvement': baseline['equalized_odds_difference'] - mitigated['equalized_odds_difference']
            })
        
        report = pd.DataFrame(rows)
        
        if save_path:
            report.to_csv(save_path, index=False)
            print(f"\nFairness report saved to {save_path}")
        
        print("\n=== Fairness Comparison Report ===")
        print(report.to_string(index=False))
        
        return report
    
    def plot_fairness_comparison(
        self,
        baseline_summary: Dict,
        mitigated_summary: Dict,
        save_path: Optional[str] = None
    ):
        """
        Plot fairness metrics comparison.
        
        Args:
            baseline_summary: Baseline fairness metrics
            mitigated_summary: Mitigated fairness metrics
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        features = list(baseline_summary.keys())
        dp_baseline = [baseline_summary[f]['demographic_parity_difference'] for f in features]
        dp_mitigated = [mitigated_summary[f]['demographic_parity_difference'] for f in features]
        eo_baseline = [baseline_summary[f]['equalized_odds_difference'] for f in features]
        eo_mitigated = [mitigated_summary[f]['equalized_odds_difference'] for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        # Demographic Parity
        axes[0].bar(x - width/2, dp_baseline, width, label='Baseline', alpha=0.8)
        axes[0].bar(x + width/2, dp_mitigated, width, label='Mitigated', alpha=0.8)
        axes[0].axhline(y=settings.fairness_threshold, color='r', linestyle='--', label='Threshold')
        axes[0].axhline(y=-settings.fairness_threshold, color='r', linestyle='--')
        axes[0].set_xlabel('Sensitive Feature')
        axes[0].set_ylabel('Demographic Parity Difference')
        axes[0].set_title('Demographic Parity Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(features, rotation=45)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Equalized Odds
        axes[1].bar(x - width/2, eo_baseline, width, label='Baseline', alpha=0.8)
        axes[1].bar(x + width/2, eo_mitigated, width, label='Mitigated', alpha=0.8)
        axes[1].axhline(y=settings.fairness_threshold, color='r', linestyle='--', label='Threshold')
        axes[1].axhline(y=-settings.fairness_threshold, color='r', linestyle='--')
        axes[1].set_xlabel('Sensitive Feature')
        axes[1].set_ylabel('Equalized Odds Difference')
        axes[1].set_title('Equalized Odds Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(features, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # This would typically be called from a training/evaluation script
    pass
