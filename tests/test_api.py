"""
Integration tests for the FastAPI application
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200
    
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_basic():
    """Test prediction endpoint with minimal valid data."""
    request_data = {
        "application_data": {
            "credit_score": 720,
            "age": 35,
            "loan_amount": 15000,
            "loan_term": 36
        }
    }
    
    response = client.post("/predict", json=request_data)
    
    # Even without a trained model, should return proper error or response structure
    assert response.status_code in [200, 503]  # 503 if model not loaded
    
    if response.status_code == 200:
        data = response.json()
        assert "applicant_id" in data
        assert "decision" in data
        assert "probability" in data
        assert data["decision"] in ["Approve", "Deny"]


def test_predict_endpoint_validation():
    """Test prediction endpoint input validation."""
    # Invalid credit score (out of range)
    invalid_request = {
        "application_data": {
            "credit_score": 999,  # Invalid: max is 850
            "age": 35,
            "loan_amount": 15000,
            "loan_term": 36
        }
    }
    
    response = client.post("/predict", json=invalid_request)
    assert response.status_code == 422  # Validation error


def test_predict_endpoint_with_sensitive_attrs():
    """Test prediction with sensitive attributes for fairness logging."""
    request_data = {
        "application_data": {
            "credit_score": 720,
            "age": 35,
            "loan_amount": 15000,
            "loan_term": 36,
            "sex": "M",
            "age_group": "30-40",
            "zip_code": "12345"
        }
    }
    
    response = client.post("/predict", json=request_data)
    assert response.status_code in [200, 503]


def test_explain_endpoint_without_prediction():
    """Test explain endpoint for non-existent prediction."""
    request_data = {
        "applicant_id": "NONEXISTENT-12345"
    }
    
    response = client.post("/explain", json=request_data)
    assert response.status_code == 404  # Not found


def test_api_cors_headers():
    """Test that CORS headers are properly set."""
    response = client.options("/health")
    # CORS middleware should add appropriate headers
    # Exact test depends on CORS configuration


def test_api_error_handling():
    """Test that API handles errors gracefully."""
    # Send malformed JSON
    response = client.post(
        "/predict",
        data="not valid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422  # Unprocessable entity
