"""
Unit tests for the IDP Engine module
"""

import pytest
import numpy as np
from pathlib import Path
from src.modules.idp_engine import IDPEngine


@pytest.fixture
def idp_engine():
    """Fixture to create IDP engine instance."""
    return IDPEngine()


def test_idp_initialization(idp_engine):
    """Test that IDP engine initializes correctly."""
    assert idp_engine is not None
    assert idp_engine.nlp is not None
    assert idp_engine.matcher is not None


def test_currency_extraction(idp_engine):
    """Test currency extraction from text."""
    test_cases = [
        ("$1,234.56", 1234.56),
        ("$1234.56", 1234.56),
        ("1234", 1234.0),
        ("$1,234", 1234.0),
    ]
    
    for text, expected in test_cases:
        result = idp_engine._extract_currency(text)
        assert result == expected, f"Expected {expected}, got {result} for '{text}'"


def test_date_extraction(idp_engine):
    """Test date extraction from text."""
    text = "Pay period: 01/01/2024 - 01/15/2024"
    dates = idp_engine._extract_dates(text)
    
    assert len(dates) == 2
    assert dates[0] == "01/01/2024"
    assert dates[1] == "01/15/2024"


def test_preprocess_image(idp_engine):
    """Test image preprocessing."""
    # Create a dummy grayscale image
    dummy_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Preprocess
    processed = idp_engine.preprocess_image(dummy_image)
    
    # Check that output is binary (only 0 or 255 values)
    assert processed.dtype == np.uint8
    assert processed.shape == dummy_image.shape
    assert set(np.unique(processed)).issubset({0, 255})


def test_extract_structured_data_paystub(idp_engine):
    """Test structured data extraction from paystub text."""
    sample_text = """
    ACME Corporation
    Pay Stub
    
    Employee: John Doe
    Pay Period: 01/01/2024 - 01/15/2024
    
    Gross Pay: $5,000.00
    Deductions: $1,000.00
    Net Pay: $4,000.00
    """
    
    result = idp_engine.extract_structured_data(sample_text, document_type='paystub')
    
    # Check that key fields were extracted
    assert 'net_income' in result or 'gross_income' in result
    assert result.get('net_income') == 4000.0 or result.get('gross_income') == 5000.0


def test_extract_structured_data_bank_statement(idp_engine):
    """Test structured data extraction from bank statement text."""
    sample_text = """
    Chase Bank
    Monthly Statement
    
    Account Balance: $10,500.00
    
    Transactions:
    01/05/2024 Direct Deposit PAYROLL $5,000.00
    01/10/2024 Auto Loan Payment -$450.00
    """
    
    result = idp_engine.extract_structured_data(sample_text, document_type='bank_statement')
    
    # Check that bank name or balance was extracted
    assert 'bank_name' in result or 'account_balance' in result
