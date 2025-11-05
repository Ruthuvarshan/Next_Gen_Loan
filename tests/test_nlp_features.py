"""
Unit tests for NLP Feature Engineering module
"""

import pytest
import pandas as pd
from src.modules.nlp_features import NLPFeatureEngine, TransactionCategorizer


@pytest.fixture
def nlp_engine():
    """Fixture to create NLP feature engine instance."""
    return NLPFeatureEngine()


@pytest.fixture
def categorizer():
    """Fixture to create transaction categorizer instance."""
    return TransactionCategorizer()


def test_categorizer_initialization(categorizer):
    """Test that categorizer initializes correctly."""
    assert categorizer is not None
    assert len(categorizer.categories) > 0
    assert 'Income' in categorizer.categories
    assert 'Debt_EMI' in categorizer.categories
    assert 'Risk_Flag' in categorizer.categories


def test_transaction_categorization(categorizer):
    """Test transaction categorization."""
    test_cases = [
        ("DIRECT DEPOSIT PAYROLL", "Income"),
        ("ACH DEPOSIT SALARY", "Income"),
        ("AUTO LOAN PAYMENT", "Debt_EMI"),
        ("MORTGAGE PMT", "Debt_EMI"),
        ("OVERDRAFT FEE", "Risk_Flag"),
        ("NSF FEE", "Risk_Flag"),
        ("STARBUCKS #1234", "Discretionary"),
        ("NETFLIX SUBSCRIPTION", "Discretionary"),
        ("PG&E UTILITY BILL", "Utility"),
        ("RANDOM STORE", "Other"),
    ]
    
    for description, expected_category in test_cases:
        result = categorizer.categorize(description)
        assert result == expected_category, f"Expected {expected_category}, got {result} for '{description}'"


def test_parse_transactions(nlp_engine):
    """Test transaction parsing from raw text."""
    sample_text = """
    08/01/2024 DIRECT DEPOSIT PAYROLL XYZ CORP $5200.00
    08/03/2024 AUTO LOAN PMT CHASE BANK -$450.00
    08/05/2024 STARBUCKS #1234 -$5.50
    08/08/2024 OVERDRAFT FEE -$35.00
    """
    
    df = nlp_engine.parse_transactions(sample_text)
    
    assert len(df) == 4
    assert 'date' in df.columns
    assert 'description' in df.columns
    assert 'amount' in df.columns
    assert 'category' in df.columns
    
    # Check categories were assigned
    assert 'Income' in df['category'].values
    assert 'Debt_EMI' in df['category'].values
    assert 'Risk_Flag' in df['category'].values


def test_calculate_income_features(nlp_engine):
    """Test income feature calculation."""
    sample_text = """
    08/01/2024 DIRECT DEPOSIT PAYROLL $5200.00
    08/15/2024 DIRECT DEPOSIT PAYROLL $5200.00
    09/01/2024 DIRECT DEPOSIT PAYROLL $5200.00
    """
    
    df = nlp_engine.parse_transactions(sample_text)
    features = nlp_engine.calculate_income_features(df)
    
    assert 'avg_salary_deposit' in features
    assert 'income_stability_variance' in features
    assert 'days_between_paychecks_mean' in features
    assert features['avg_salary_deposit'] == 5200.0
    assert features['num_income_deposits'] == 3


def test_calculate_risk_features(nlp_engine):
    """Test risk feature calculation."""
    sample_text = """
    08/01/2024 DIRECT DEPOSIT PAYROLL $5200.00
    08/05/2024 OVERDRAFT FEE -$35.00
    08/10/2024 NSF FEE -$30.00
    08/15/2024 DIRECT DEPOSIT PAYROLL $5200.00
    """
    
    df = nlp_engine.parse_transactions(sample_text)
    features = nlp_engine.calculate_risk_features(df)
    
    assert 'risk_flag_count' in features
    assert 'avg_daily_balance' in features
    assert features['risk_flag_count'] == 2


def test_extract_features_complete(nlp_engine):
    """Test complete feature extraction pipeline."""
    sample_text = """
    08/01/2024 DIRECT DEPOSIT PAYROLL $5200.00
    08/03/2024 AUTO LOAN PMT -$450.00
    08/05/2024 STARBUCKS #1234 -$5.50
    08/08/2024 OVERDRAFT FEE -$35.00
    08/15/2024 DIRECT DEPOSIT PAYROLL $5200.00
    08/20/2024 NETFLIX SUBSCRIPTION -$15.99
    """
    
    features = nlp_engine.extract_features(sample_text)
    
    # Check that all feature groups are present
    assert 'avg_salary_deposit' in features
    assert 'monthly_emi_total' in features
    assert 'risk_flag_count' in features
    assert 'utilization_ratio_proxy' in features
    assert 'num_transactions' in features
    
    # Check that values are reasonable
    assert features['avg_salary_deposit'] == 5200.0
    assert features['risk_flag_count'] == 1
    assert features['num_transactions'] == 6
