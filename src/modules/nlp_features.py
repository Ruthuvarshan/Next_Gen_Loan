"""
Module 2: NLP Feature Engineering & Risk Primitives

This module transforms raw bank transaction text into high-signal 
behavioral risk features through:
1. Transaction parsing and categorization
2. Temporal aggregation and pattern detection
3. Risk primitive generation (income stability, debt ratios, etc.)
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class TransactionCategorizer:
    """Categorizes bank transactions into financial behavior categories."""
    
    def __init__(self):
        """Initialize with category patterns."""
        self.categories = {
            'Income': [
                r'payroll', r'salary', r'direct\s*dep', r'ach\s*deposit',
                r'govt\s*benefit', r'social\s*security', r'unemployment',
                r'dividend', r'interest\s*income', r'refund', r'reimbursement'
            ],
            'Debt_EMI': [
                r'loan\s*pmt', r'mortgage', r'auto\s*loan', r'car\s*payment',
                r'klarna', r'afterpay', r'affirm', r'credit\s*card\s*pmt',
                r'student\s*loan', r'personal\s*loan', r'lease\s*payment'
            ],
            'Utility': [
                r'pg&e', r'pacific\s*gas', r'electric', r'water\s*dept',
                r'gas\s*company', r'comcast', r'att\s*bill', r'verizon',
                r'internet', r'phone\s*bill', r'cable', r'utilities'
            ],
            'Discretionary': [
                r'starbucks', r'restaurant', r'coffee', r'dining',
                r'netflix', r'spotify', r'amazon', r'entertainment',
                r'shopping', r'retail', r'grocery', r'food', r'uber',
                r'lyft', r'travel', r'hotel', r'airline'
            ],
            'Risk_Flag': [
                r'overdraft', r'nsf', r'insufficient\s*funds', r'late\s*fee',
                r'penalty', r'returned\s*check', r'payday\s*loan',
                r'cash\s*advance', r'collection', r'garnishment'
            ]
        }
        
        # Compile regex patterns
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) 
                      for pattern in patterns]
            for category, patterns in self.categories.items()
        }
    
    def categorize(self, description: str) -> str:
        """
        Categorize a transaction based on its description.
        
        Args:
            description: Transaction description text
            
        Returns:
            Category name or 'Other'
        """
        description = description.lower()
        
        # Check each category
        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(description):
                    return category
        
        return 'Other'


class NLPFeatureEngine:
    """
    Extracts behavioral risk features from bank statement transactions.
    """
    
    def __init__(self):
        """Initialize the NLP feature engine."""
        self.categorizer = TransactionCategorizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50,
            ngram_range=(1, 2),
            stop_words='english'
        )
    
    def parse_transactions(self, raw_text: str) -> pd.DataFrame:
        """
        Parse raw bank statement text into structured transactions.
        
        Args:
            raw_text: Raw text from bank statement
            
        Returns:
            DataFrame with columns: date, description, amount, category
        """
        transactions = []
        
        # Transaction line pattern: DATE DESCRIPTION AMOUNT
        # Example: "08/01/2024 TFR FRM PAYPAL *JOHN DOE $150.00"
        pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(.+?)\s+(\$?-?\d+[,\d]*\.?\d*)'
        
        matches = re.finditer(pattern, raw_text)
        
        for match in matches:
            date_str, description, amount_str = match.groups()
            
            # Parse date
            try:
                date = pd.to_datetime(date_str)
            except:
                continue
            
            # Parse amount
            amount = float(re.sub(r'[\$,]', '', amount_str))
            
            # Categorize transaction
            category = self.categorizer.categorize(description)
            
            transactions.append({
                'date': date,
                'description': description.strip(),
                'amount': amount,
                'category': category
            })
        
        df = pd.DataFrame(transactions)
        
        if not df.empty:
            df = df.sort_values('date')
        
        return df
    
    def calculate_income_features(self, df: pd.DataFrame) -> Dict:
        """
        Calculate income stability and pattern features.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            Dictionary of income features
        """
        income_txns = df[df['category'] == 'Income']
        
        if income_txns.empty:
            return {
                'avg_salary_deposit': 0.0,
                'income_stability_variance': 0.0,
                'days_between_paychecks_mean': 0.0,
                'days_between_paychecks_std': 0.0,
                'num_income_deposits': 0
            }
        
        # Average income deposit
        avg_salary = income_txns['amount'].mean()
        
        # Income variance (stability metric)
        income_variance = income_txns['amount'].std()
        
        # Days between paychecks
        income_txns = income_txns.sort_values('date')
        days_between = income_txns['date'].diff().dt.days.dropna()
        
        days_between_mean = days_between.mean() if not days_between.empty else 0
        days_between_std = days_between.std() if not days_between.empty else 0
        
        return {
            'avg_salary_deposit': float(avg_salary),
            'income_stability_variance': float(income_variance),
            'days_between_paychecks_mean': float(days_between_mean),
            'days_between_paychecks_std': float(days_between_std),
            'num_income_deposits': len(income_txns)
        }
    
    def calculate_debt_features(self, df: pd.DataFrame) -> Dict:
        """
        Calculate debt and affordability features.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            Dictionary of debt features
        """
        debt_txns = df[df['category'] == 'Debt_EMI']
        
        if debt_txns.empty:
            return {
                'monthly_emi_total': 0.0,
                'num_debt_payments': 0,
                'avg_debt_payment': 0.0
            }
        
        # Calculate monthly EMI total
        # Assuming statement covers one month
        monthly_emi = abs(debt_txns['amount'].sum())
        
        return {
            'monthly_emi_total': float(monthly_emi),
            'num_debt_payments': len(debt_txns),
            'avg_debt_payment': float(debt_txns['amount'].mean())
        }
    
    def calculate_spending_features(self, df: pd.DataFrame) -> Dict:
        """
        Calculate discretionary spending features.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            Dictionary of spending features
        """
        discretionary_txns = df[df['category'] == 'Discretionary']
        utility_txns = df[df['category'] == 'Utility']
        
        return {
            'avg_discretionary_spend': float(abs(discretionary_txns['amount'].sum())),
            'avg_utility_spend': float(abs(utility_txns['amount'].sum())),
            'num_discretionary_txns': len(discretionary_txns)
        }
    
    def calculate_risk_features(self, df: pd.DataFrame) -> Dict:
        """
        Calculate behavioral risk flags and financial distress indicators.
        
        Args:
            df: DataFrame of transactions
            
        Returns:
            Dictionary of risk features
        """
        risk_txns = df[df['category'] == 'Risk_Flag']
        
        # Calculate daily balance proxy (running sum)
        df_sorted = df.sort_values('date')
        df_sorted['running_balance'] = df_sorted['amount'].cumsum()
        
        # Average daily balance (simplified)
        avg_daily_balance = df_sorted['running_balance'].mean()
        min_balance = df_sorted['running_balance'].min()
        
        # Count months with zero overdrafts
        df_sorted['month'] = df_sorted['date'].dt.to_period('M')
        risk_by_month = df_sorted[df_sorted['category'] == 'Risk_Flag'].groupby('month').size()
        total_months = df_sorted['month'].nunique()
        months_with_zero_overdraft = total_months - len(risk_by_month)
        
        # Check for payday loans
        payday_loan_patterns = ['payday', 'cash advance', 'fast cash']
        relies_on_payday_loans = any(
            any(pattern in desc.lower() for pattern in payday_loan_patterns)
            for desc in df['description']
        )
        
        return {
            'risk_flag_count': len(risk_txns),
            'avg_daily_balance': float(avg_daily_balance),
            'min_balance': float(min_balance),
            'months_with_zero_overdraft': int(months_with_zero_overdraft),
            'relies_on_payday_loans': int(relies_on_payday_loans)
        }
    
    def calculate_composite_features(self, features: Dict) -> Dict:
        """
        Calculate composite features from basic features.
        
        Args:
            features: Dictionary of basic features
            
        Returns:
            Dictionary of composite features
        """
        # Utilization ratio proxy: (EMI + discretionary) / income
        denominator = features.get('avg_salary_deposit', 1.0)
        if denominator == 0:
            denominator = 1.0
        
        total_expenses = (
            features.get('monthly_emi_total', 0) +
            features.get('avg_discretionary_spend', 0)
        )
        
        utilization_ratio = total_expenses / denominator
        
        # Financial health score (custom composite)
        financial_health = (
            (features.get('months_with_zero_overdraft', 0) * 10) -
            (features.get('risk_flag_count', 0) * 5) +
            (features.get('avg_daily_balance', 0) / 100)
        )
        
        return {
            'utilization_ratio_proxy': float(utilization_ratio),
            'financial_health_score': float(financial_health)
        }
    
    def vectorize_loan_purpose(self, purpose_text: str, fit: bool = False) -> np.ndarray:
        """
        Vectorize loan purpose text using TF-IDF.
        
        Args:
            purpose_text: Loan purpose description
            fit: Whether to fit the vectorizer
            
        Returns:
            TF-IDF feature vector
        """
        if fit:
            vector = self.tfidf_vectorizer.fit_transform([purpose_text])
        else:
            vector = self.tfidf_vectorizer.transform([purpose_text])
        
        return vector.toarray()[0]
    
    def extract_features(
        self,
        bank_statement_text: str,
        loan_purpose: Optional[str] = None
    ) -> Dict:
        """
        Complete feature extraction pipeline.
        
        Args:
            bank_statement_text: Raw text from bank statement
            loan_purpose: Optional loan purpose text
            
        Returns:
            Dictionary of all extracted features
        """
        # Parse transactions
        df = self.parse_transactions(bank_statement_text)
        
        if df.empty:
            return {
                'error': 'No transactions found',
                'feature_count': 0
            }
        
        # Extract feature groups
        income_features = self.calculate_income_features(df)
        debt_features = self.calculate_debt_features(df)
        spending_features = self.calculate_spending_features(df)
        risk_features = self.calculate_risk_features(df)
        
        # Combine all features
        all_features = {
            **income_features,
            **debt_features,
            **spending_features,
            **risk_features
        }
        
        # Add composite features
        composite_features = self.calculate_composite_features(all_features)
        all_features.update(composite_features)
        
        # Add metadata
        all_features['num_transactions'] = len(df)
        all_features['statement_date_range_days'] = (
            (df['date'].max() - df['date'].min()).days
            if not df.empty else 0
        )
        
        return all_features


# Example usage
if __name__ == "__main__":
    # Initialize NLP engine
    nlp_engine = NLPFeatureEngine()
    
    # Sample bank statement text
    sample_text = """
    08/01/2024 DIRECT DEPOSIT PAYROLL XYZ CORP $5200.00
    08/03/2024 AUTO LOAN PMT CHASE BANK -$450.00
    08/05/2024 STARBUCKS #1234 -$5.50
    08/08/2024 OVERDRAFT FEE -$35.00
    08/15/2024 DIRECT DEPOSIT PAYROLL XYZ CORP $5200.00
    08/20/2024 NETFLIX SUBSCRIPTION -$15.99
    """
    
    # Extract features
    features = nlp_engine.extract_features(sample_text)
    
    print("Extracted NLP Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
