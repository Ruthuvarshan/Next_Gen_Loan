"""
Module 4: Explainable AI (XAI) and Adverse Action Generator

This module provides:
1. Global model interpretation using SHAP
2. Local prediction explanations (waterfall plots)
3. Automated adverse action reason code generation (ECOA compliance)
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from src.utils.config import settings


# Mapping from internal feature names to human-readable adverse action reasons
REASON_CODE_MAP = {
    'risk_flag_count': 'Frequent instances of overdrafts, non-sufficient funds, or late fees',
    'income_stability_variance': 'Irregular or unverifiable income deposits',
    'utilization_ratio_proxy': 'High ratio of monthly expenses to verified income',
    'credit_score': 'Credit score does not meet minimum requirements',
    'monthly_emi_total': 'High level of existing debt obligations',
    'loan_to_verified_income': 'Requested loan amount is too high relative to verified income',
    'relies_on_payday_loans': 'Recent use of high-risk payday loan or cash advance services',
    'months_with_zero_overdraft': 'Insufficient history of maintaining positive account balance',
    'min_balance': 'Insufficient available funds or negative account balance',
    'days_between_paychecks_std': 'Inconsistent employment or income timing',
    'num_debt_payments': 'High number of existing debt obligations',
    'financial_health_score': 'Overall financial health indicators do not meet requirements',
    'age': 'Length of credit history does not meet minimum requirements',
    'loan_term': 'Requested loan term exceeds acceptable limits',
    'loan_amount': 'Requested loan amount exceeds lending limits'
}


class XAIExplainer:
    """
    SHAP-based explainer for credit risk model predictions.
    Provides both global and local interpretability.
    """
    
    def __init__(self, model, X_background: Optional[pd.DataFrame] = None):
        """
        Initialize the XAI explainer.
        
        Args:
            model: Trained XGBoost model
            X_background: Background dataset for SHAP (sample of training data)
        """
        self.model = model
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        
        # Initialize SHAP explainer
        if X_background is not None:
            self.initialize_explainer(X_background)
    
    def initialize_explainer(self, X_background: pd.DataFrame):
        """
        Initialize the SHAP TreeExplainer with background data.
        
        Args:
            X_background: Background dataset (typically a sample of training data)
        """
        # Use TreeExplainer for XGBoost (much faster than KernelExplainer)
        self.explainer = shap.TreeExplainer(
            self.model,
            X_background
        )
        
        self.feature_names = X_background.columns.tolist()
        
        print(f"SHAP explainer initialized with {len(X_background)} background samples")
    
    def calculate_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate SHAP values for the given dataset.
        
        Args:
            X: Dataset to explain
            
        Returns:
            SHAP values array
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Call initialize_explainer() first.")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Store for later use
        self.shap_values = shap_values
        
        return shap_values
    
    def plot_global_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = 'beeswarm',
        max_display: int = 20,
        save_path: Optional[str] = None
    ):
        """
        Generate global SHAP summary plot (beeswarm or bar).
        
        This shows which features are most important globally and 
        how their values impact predictions.
        
        Args:
            X: Dataset to explain
            plot_type: Type of plot ('beeswarm', 'bar', or 'violin')
            max_display: Maximum number of features to display
            save_path: Optional path to save the plot
        """
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            shap_values = self.calculate_shap_values(X)
        else:
            shap_values = self.shap_values
        
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'beeswarm':
            shap.plots.beeswarm(
                shap.Explanation(
                    values=shap_values,
                    base_values=self.explainer.expected_value,
                    data=X.values,
                    feature_names=X.columns.tolist()
                ),
                max_display=max_display,
                show=False
            )
        elif plot_type == 'bar':
            shap.plots.bar(
                shap.Explanation(
                    values=shap_values,
                    base_values=self.explainer.expected_value,
                    data=X.values,
                    feature_names=X.columns.tolist()
                ),
                max_display=max_display,
                show=False
            )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_waterfall(
        self,
        X: pd.DataFrame,
        instance_index: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Generate waterfall plot for a single prediction.
        
        This shows how each feature contributed to moving the prediction
        from the base value to the final output.
        
        Args:
            X: Dataset containing the instance to explain
            instance_index: Index of the instance to explain
            save_path: Optional path to save the plot
        """
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            shap_values = self.calculate_shap_values(X)
        else:
            shap_values = self.shap_values
        
        plt.figure(figsize=(10, 6))
        
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[instance_index],
                base_values=self.explainer.expected_value,
                data=X.iloc[instance_index].values,
                feature_names=X.columns.tolist()
            ),
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_force(
        self,
        X: pd.DataFrame,
        instance_index: int = 0,
        matplotlib: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Generate force plot for a single prediction.
        
        Args:
            X: Dataset containing the instance to explain
            instance_index: Index of the instance to explain
            matplotlib: Whether to use matplotlib (vs JavaScript)
            save_path: Optional path to save the plot
        """
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            shap_values = self.calculate_shap_values(X)
        else:
            shap_values = self.shap_values
        
        if matplotlib:
            shap.plots.force(
                self.explainer.expected_value,
                shap_values[instance_index],
                X.iloc[instance_index],
                matplotlib=True,
                show=False
            )
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        else:
            # Interactive JavaScript plot
            shap.plots.force(
                self.explainer.expected_value,
                shap_values[instance_index],
                X.iloc[instance_index]
            )
    
    def generate_adverse_action_reasons(
        self,
        shap_values_instance: np.ndarray,
        feature_names: List[str],
        feature_values: pd.Series,
        top_n: int = 3
    ) -> List[str]:
        """
        Generate human-readable adverse action reasons for a denied application.
        
        This function is ECOA-compliant and generates the specific reasons
        that must be provided to applicants whose loans are denied.
        
        Args:
            shap_values_instance: SHAP values for a single instance
            feature_names: List of feature names
            feature_values: Actual feature values for the instance
            top_n: Number of reasons to return
            
        Returns:
            List of human-readable denial reasons
        """
        # Zip features with their SHAP contributions
        contribs = list(zip(feature_names, shap_values_instance, feature_values))
        
        # Filter for negative contributions (features that hurt the score)
        # Assuming positive SHAP = toward approval, negative SHAP = toward denial
        negative_contribs = [c for c in contribs if c[1] < 0]
        
        # Sort by magnitude (most negative impact first)
        negative_contribs.sort(key=lambda x: x[1])
        
        # Map the top_n features to their human-readable reasons
        reasons = []
        for feature, shap_val, value in negative_contribs[:top_n]:
            if feature in REASON_CODE_MAP:
                reason = REASON_CODE_MAP[feature]
                # Optionally add the actual value for context
                reasons.append(f"{reason} (Value: {value:.2f}, Impact: {shap_val:.4f})")
            else:
                # Fallback for unmapped features
                reasons.append(f"Feature '{feature}' does not meet requirements (Value: {value:.2f})")
        
        # If we don't have enough reasons, add a generic one
        if len(reasons) < top_n:
            reasons.append("Overall risk profile does not meet lending criteria")
        
        return reasons
    
    def explain_prediction(
        self,
        X: pd.DataFrame,
        instance_index: int,
        decision: str,
        probability: float,
        generate_reasons: bool = True
    ) -> Dict:
        """
        Complete explanation package for a single prediction.
        
        Args:
            X: Dataset containing the instance
            instance_index: Index of the instance
            decision: Model decision ('Approve' or 'Deny')
            probability: Prediction probability
            generate_reasons: Whether to generate adverse action reasons
            
        Returns:
            Dictionary with explanation components
        """
        # Calculate SHAP values
        if self.shap_values is None:
            shap_values = self.calculate_shap_values(X)
        else:
            shap_values = self.shap_values
        
        instance_shap = shap_values[instance_index]
        
        # Get feature contributions
        contributions = pd.DataFrame({
            'feature': X.columns.tolist(),
            'value': X.iloc[instance_index].values,
            'shap_value': instance_shap
        }).sort_values('shap_value', ascending=True)
        
        explanation = {
            'decision': decision,
            'probability': probability,
            'base_value': float(self.explainer.expected_value),
            'contributions': contributions.to_dict('records'),
            'top_positive_features': contributions.tail(5).to_dict('records'),
            'top_negative_features': contributions.head(5).to_dict('records')
        }
        
        # Generate adverse action reasons if denied
        if generate_reasons and decision == 'Deny':
            reasons = self.generate_adverse_action_reasons(
                instance_shap,
                X.columns.tolist(),
                X.iloc[instance_index],
                top_n=settings.shap_top_n_reasons
            )
            explanation['adverse_action_reasons'] = reasons
        
        return explanation
    
    def batch_explain(
        self,
        X: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate explanations for a batch of predictions.
        
        Args:
            X: Dataset to explain
            predictions: Model predictions
            probabilities: Prediction probabilities
            threshold: Classification threshold
            
        Returns:
            DataFrame with explanations
        """
        # Calculate SHAP values
        shap_values = self.calculate_shap_values(X)
        
        explanations = []
        
        for i in range(len(X)):
            decision = 'Deny' if predictions[i] == 1 else 'Approve'
            prob = probabilities[i]
            
            explanation = self.explain_prediction(
                X,
                i,
                decision,
                prob,
                generate_reasons=(decision == 'Deny')
            )
            
            explanations.append({
                'index': i,
                'decision': decision,
                'probability': prob,
                'top_negative_feature': explanation['top_negative_features'][0]['feature'] if explanation['top_negative_features'] else None,
                'top_negative_impact': explanation['top_negative_features'][0]['shap_value'] if explanation['top_negative_features'] else None
            })
        
        return pd.DataFrame(explanations)


def get_simple_adverse_action_reasons(
    shap_values_instance: np.ndarray,
    feature_names: List[str],
    top_n: int = 3
) -> List[str]:
    """
    Simplified version of adverse action reason generator for API use.
    
    Args:
        shap_values_instance: SHAP values for one applicant
        feature_names: List of feature names
        top_n: Number of reasons to return
        
    Returns:
        List of human-readable denial reasons
    """
    # Zip features and their SHAP contributions
    contribs = list(zip(feature_names, shap_values_instance))
    
    # Filter for negative contributions
    negative_contribs = [c for c in contribs if c[1] < 0]
    
    # Sort by magnitude (most negative impact first)
    negative_contribs.sort(key=lambda x: x[1])
    
    # Map to human-readable reasons
    reasons = []
    for feature, shap_val in negative_contribs[:top_n]:
        if feature in REASON_CODE_MAP:
            reasons.append(REASON_CODE_MAP[feature])
        else:
            reasons.append(f"Feature '{feature}' does not meet requirements")
    
    return reasons


# Example usage
if __name__ == "__main__":
    # This would typically be called from the main application
    pass
