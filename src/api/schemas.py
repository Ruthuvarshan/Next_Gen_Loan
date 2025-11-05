"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class TraditionalApplicationData(BaseModel):
    """Traditional loan application data."""
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    loan_term: int = Field(..., gt=0, description="Loan term in months")
    annual_income: Optional[float] = Field(None, ge=0, description="Self-reported annual income")
    employment_length: Optional[int] = Field(None, ge=0, description="Years at current employer")
    loan_purpose: Optional[str] = Field(None, description="Purpose of the loan")
    
    # Sensitive attributes (for fairness monitoring only, not used in prediction)
    sex: Optional[str] = Field(None, description="Sex (for fairness audit only)")
    age_group: Optional[str] = Field(None, description="Age group (for fairness audit only)")
    zip_code: Optional[str] = Field(None, description="ZIP code (proxy for demographics)")


class PredictionRequest(BaseModel):
    """Request for loan prediction."""
    applicant_id: Optional[str] = Field(None, description="Unique applicant identifier")
    application_data: TraditionalApplicationData
    paystub_text: Optional[str] = Field(None, description="Extracted paystub text (if already processed)")
    bank_statement_text: Optional[str] = Field(None, description="Extracted bank statement text")


class FeatureContribution(BaseModel):
    """SHAP feature contribution."""
    feature: str
    value: float
    shap_value: float


class PredictionResponse(BaseModel):
    """Response for loan prediction."""
    applicant_id: str
    decision: str = Field(..., description="Approve or Deny")
    probability: float = Field(..., ge=0, le=1, description="Probability of default")
    confidence: str = Field(..., description="High, Medium, or Low confidence")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "applicant_id": "APP-2024-001",
                "decision": "Approve",
                "probability": 0.15,
                "confidence": "High",
                "timestamp": "2024-11-04T10:30:00Z"
            }
        }


class ExplanationRequest(BaseModel):
    """Request for prediction explanation."""
    applicant_id: str = Field(..., description="Unique applicant identifier")


class ExplanationResponse(BaseModel):
    """Response with prediction explanation and adverse action reasons."""
    applicant_id: str
    decision: str
    probability: float
    base_value: float
    top_positive_features: List[FeatureContribution]
    top_negative_features: List[FeatureContribution]
    adverse_action_reasons: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "applicant_id": "APP-2024-002",
                "decision": "Deny",
                "probability": 0.75,
                "base_value": 0.50,
                "top_positive_features": [
                    {"feature": "credit_score", "value": 720, "shap_value": 0.08}
                ],
                "top_negative_features": [
                    {"feature": "risk_flag_count", "value": 8, "shap_value": -0.15}
                ],
                "adverse_action_reasons": [
                    "Frequent instances of overdrafts, non-sufficient funds, or late fees",
                    "High ratio of monthly expenses to verified income"
                ],
                "timestamp": "2024-11-04T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_loaded: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
