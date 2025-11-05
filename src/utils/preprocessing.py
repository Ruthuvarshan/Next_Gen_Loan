"""
Preprocessing utilities for data transformation and validation.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """Handles data preprocessing and validation for model input."""
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important financial symbols
        text = re.sub(r'[^a-zA-Z0-9\s\$\.\,\-\/]', '', text)
        return text.strip()
    
    def parse_currency(self, value: str) -> float:
        """Parse currency string to float."""
        if isinstance(value, (int, float)):
            return float(value)
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[\$\,]', '', str(value))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object."""
        date_formats = [
            '%Y-%m-%d',
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y/%m/%d',
            '%b %d, %Y',
            '%B %d, %Y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None
    
    def validate_features(self, features: Dict) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present and valid.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        required_features = [
            'credit_score',
            'loan_amount',
            'loan_term',
            'age'
        ]
        
        errors = []
        
        for feature in required_features:
            if feature not in features:
                errors.append(f"Missing required feature: {feature}")
            elif features[feature] is None:
                errors.append(f"Feature {feature} is None")
        
        # Validate ranges
        if 'credit_score' in features:
            score = features['credit_score']
            if not (300 <= score <= 850):
                errors.append(f"Credit score {score} out of valid range [300, 850]")
        
        if 'age' in features:
            age = features['age']
            if not (18 <= age <= 100):
                errors.append(f"Age {age} out of valid range [18, 100]")
        
        if 'loan_amount' in features:
            amount = features['loan_amount']
            if amount <= 0:
                errors.append(f"Loan amount must be positive, got {amount}")
        
        return len(errors) == 0, errors
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features for improved model performance."""
        df = df.copy()
        
        # Loan-to-income ratio
        if 'loan_amount' in df.columns and 'verified_net_income' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amount'] / (df['verified_net_income'] * 12 + 1)
        
        # Payment-to-income ratio
        if 'loan_amount' in df.columns and 'loan_term' in df.columns and 'verified_net_income' in df.columns:
            monthly_payment = df['loan_amount'] / df['loan_term']
            df['payment_to_income_ratio'] = monthly_payment / (df['verified_net_income'] + 1)
        
        # Risk score composite
        if 'credit_score' in df.columns and 'risk_flag_count' in df.columns:
            df['risk_score_composite'] = df['credit_score'] / (df['risk_flag_count'] + 1)
        
        # Stability score
        if 'income_stability_variance' in df.columns and 'months_with_zero_overdraft' in df.columns:
            df['financial_stability_score'] = (
                df['months_with_zero_overdraft'] / (df['income_stability_variance'] + 1)
            )
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        df = df.copy()
        
        # Numeric features: fill with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical features: fill with mode or 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna('Unknown', inplace=True)
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input dataframe
            fit: If True, fit the scaler; otherwise use existing scaler
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            self.scaler = StandardScaler()
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        return df
