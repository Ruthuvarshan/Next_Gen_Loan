"""
Module 3: XGBoost Credit Risk Model

This module handles:
1. Feature matrix assembly from multiple sources
2. SMOTE for handling class imbalance
3. XGBoost model training with hyperparameter tuning
4. Comprehensive model evaluation
"""
from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import settings


class CreditRiskModel:
    """
    XGBoost-based credit risk prediction model with SMOTE handling
    and comprehensive evaluation.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the credit risk model.
        
        Args:
            model_path: Path to load existing model from
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: Optional[List[str]] = None
        self.smote: Optional[SMOTE] = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def assemble_feature_matrix(
        self,
        traditional_data: pd.DataFrame,
        idp_data: Optional[pd.DataFrame] = None,
        nlp_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Assemble the complete feature matrix from multiple sources.
        
        Sources:
        1. Traditional: Credit score, age, loan amount, etc.
        2. IDP-Extracted: Verified income from documents
        3. NLP-Generated: Behavioral risk features
        4. Engineered: Interaction features
        
        Args:
            traditional_data: Traditional application data
            idp_data: IDP-extracted data
            nlp_features: NLP-generated features
            
        Returns:
            Complete feature matrix
        """
        # Start with traditional data
        feature_matrix = traditional_data.copy()
        
        # Merge IDP data if available
        if idp_data is not None:
            feature_matrix = feature_matrix.merge(
                idp_data,
                left_index=True,
                right_index=True,
                how='left'
            )
        
        # Merge NLP features if available
        if nlp_features is not None:
            feature_matrix = feature_matrix.merge(
                nlp_features,
                left_index=True,
                right_index=True,
                how='left'
            )
        
        # Generate interaction features
        feature_matrix = self._create_interaction_features(feature_matrix)
        
        # Store feature names
        self.feature_names = feature_matrix.columns.tolist()
        
        return feature_matrix
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features for improved predictive power."""
        df = df.copy()
        
        # Loan-to-income ratio
        if 'loan_amount' in df.columns and 'verified_net_income' in df.columns:
            df['loan_to_verified_income'] = (
                df['loan_amount'] / (df['verified_net_income'] * 12 + 1)
            )
        
        # Credit score × risk flags interaction
        if 'credit_score' in df.columns and 'risk_flag_count' in df.columns:
            df['credit_risk_interaction'] = (
                df['credit_score'] / (df['risk_flag_count'] + 1)
            )
        
        # Income stability × debt ratio
        if all(col in df.columns for col in ['income_stability_variance', 'monthly_emi_total', 'avg_salary_deposit']):
            df['stability_debt_ratio'] = (
                (df['income_stability_variance'] + 1) * 
                (df['monthly_emi_total'] / (df['avg_salary_deposit'] + 1))
            )
        
        return df
    
    def handle_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sampling_strategy: str = 'auto'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to handle class imbalance in training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            sampling_strategy: SMOTE sampling strategy
            
        Returns:
            Resampled X_train and y_train
        """
        self.smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=5
        )
        
        X_resampled, y_resampled = self.smote.fit_resample(X_train, y_train)
        
        print(f"Original class distribution: {dict(pd.Series(y_train).value_counts())}")
        print(f"Resampled class distribution: {dict(pd.Series(y_resampled).value_counts())}")
        
        return X_resampled, y_resampled
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        use_smote: bool = True,
        hyperparameter_tuning: bool = False,
        cv_folds: int = 5
    ) -> Dict:
        """
        Train the XGBoost model with optional SMOTE and hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to apply SMOTE
            hyperparameter_tuning: Whether to perform GridSearchCV
            cv_folds: Number of cross-validation folds
            
        Returns:
            Training metrics dictionary
        """
        # Handle class imbalance with SMOTE
        if use_smote:
            X_train_balanced, y_train_balanced = self.handle_imbalance(X_train, y_train)
        else:
            X_train_balanced = X_train
            y_train_balanced = y_train
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1])
        
        if hyperparameter_tuning:
            # Perform hyperparameter tuning
            self.model = self._hyperparameter_tuning(
                X_train_balanced,
                y_train_balanced,
                scale_pos_weight,
                cv_folds
            )
        else:
            # Use default parameters with scale_pos_weight
            self.model = xgb.XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                objective='binary:logistic',
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric='auc',
                use_label_encoder=False
            )
            
            self.model.fit(X_train_balanced, y_train_balanced)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.model,
            X_train_balanced,
            y_train_balanced,
            cv=cv_folds,
            scoring='roc_auc'
        )
        
        training_metrics = {
            'cv_mean_auc': cv_scores.mean(),
            'cv_std_auc': cv_scores.std(),
            'training_samples': len(X_train_balanced)
        }
        
        print(f"Cross-validation AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return training_metrics
    
    def _hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        scale_pos_weight: float,
        cv_folds: int
    ) -> xgb.XGBClassifier:
        """
        Perform GridSearchCV for hyperparameter optimization.
        
        Args:
            X: Training features
            y: Training labels
            scale_pos_weight: Weight for minority class
            cv_folds: Number of CV folds
            
        Returns:
            Best estimator from grid search
        """
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='auc',
            use_label_encoder=False
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get predictions
        y_pred, y_prob = self.predict(X_test)
        
        # Apply custom threshold if specified
        if threshold != 0.5:
            y_pred = (y_prob >= threshold).astype(int)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # False positive rate and false negative rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'threshold': threshold
        }
        
        print("\n=== Model Evaluation ===")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")
        print(f"False Negative Rate: {fnr:.4f}")
        print("\nConfusion Matrix:")
        print(f"TN: {tn:4d}  FP: {fp:4d}")
        print(f"FN: {fn:4d}  TP: {tp:4d}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix heatmap."""
        y_pred, _ = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Approve', 'Deny'],
            yticklabels=['Approve', 'Deny']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        save_path: Optional[str] = None
    ):
        """Plot ROC curve."""
        _, y_prob = self.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_model(self, model_path: Union[str, Path]):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'smote': self.smote
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Union[str, Path]):
        """Load a trained model from disk."""
        model_data = joblib.load(model_path)
        
        # Handle both old format (dict) and new format (sklearn Pipeline)
        if isinstance(model_data, dict):
            # Old format: dictionary with model, feature_names, smote
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.smote = model_data.get('smote', None)
        else:
            # New format: sklearn Pipeline from train_simple_model.py
            self.model = model_data  # The pipeline itself
            # Extract feature names from the pipeline's preprocessor
            try:
                preprocessor = model_data.named_steps['preprocessor']
                # Get feature names from the transformers
                feature_names = []
                
                for name, transformer, columns in preprocessor.transformers_:
                    if name == 'num':
                        # Numeric columns - use as is
                        feature_names.extend(columns)
                    elif name == 'cat':
                        # Categorical columns - get one-hot encoded names
                        if hasattr(transformer.named_steps['ohe'], 'get_feature_names_out'):
                            cat_features = transformer.named_steps['ohe'].get_feature_names_out(columns)
                            feature_names.extend(cat_features)
                        else:
                            # Fallback for older sklearn
                            feature_names.extend(columns)
                
                self.feature_names = feature_names
            except Exception as e:
                print(f"Warning: Could not extract feature names from pipeline: {e}")
                self.feature_names = None
            
            self.smote = None  # Pipeline doesn't use SMOTE separately
        
        print(f"Model loaded from {model_path}")


# Example usage
if __name__ == "__main__":
    # This would typically be called from a training script
    pass
