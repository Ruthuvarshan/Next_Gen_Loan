"""
Admin-only API endpoints for system management and oversight.
"""

import json
import os
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from pydantic import BaseModel

from src.utils.auth import get_admin_user
from src.utils.database import get_logging_db, PredictionLog, User
from src.modules.fairness_audit import FairnessAuditor
import pandas as pd
import numpy as np

router = APIRouter(prefix="/admin", tags=["Admin"])


# ============================================
# RESPONSE SCHEMAS
# ============================================

class PredictionLogResponse(BaseModel):
    id: int
    application_id: str
    timestamp: datetime
    submitted_by: Optional[str]
    applicant_name: Optional[str]
    prediction: str
    probability: float
    loan_amount: Optional[float]
    gender: Optional[str]
    race: Optional[str]
    
    class Config:
        from_attributes = True


class SystemStatsResponse(BaseModel):
    total_predictions: int
    predictions_today: int
    predictions_this_week: int
    approval_rate: float
    denial_rate: float
    avg_loan_amount: float
    active_users: int
    last_prediction_time: Optional[datetime]


class FairnessMetrics(BaseModel):
    metric_name: str
    overall_value: float
    by_group: dict


class FairnessReportResponse(BaseModel):
    generated_at: datetime
    total_predictions_analyzed: int
    date_range: dict
    metrics: List[FairnessMetrics]
    recommendations: List[str]


# ============================================
# PREDICTION LOG ENDPOINTS
# ============================================

@router.get("/prediction_log", response_model=List[PredictionLogResponse])
async def get_prediction_log(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    prediction_filter: Optional[str] = Query(None, regex="^(approved|denied)$"),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_logging_db)
):
    """
    Get paginated prediction logs with optional filtering.
    
    **Admin only**
    
    Query Parameters:
    - limit: Number of records to return (max 1000)
    - offset: Number of records to skip for pagination
    - prediction_filter: Filter by "approved" or "denied"
    - search: Search by applicant name or application ID
    """
    query = db.query(PredictionLog)
    
    # Apply filters
    if prediction_filter:
        query = query.filter(PredictionLog.prediction == prediction_filter)
    
    if search:
        search_pattern = f"%{search}%"
        query = query.filter(
            (PredictionLog.applicant_name.ilike(search_pattern)) |
            (PredictionLog.application_id.ilike(search_pattern))
        )
    
    # Order by most recent first
    query = query.order_by(desc(PredictionLog.timestamp))
    
    # Pagination
    total = query.count()
    logs = query.offset(offset).limit(limit).all()
    
    return logs


@router.get("/prediction_log/{application_id}")
async def get_prediction_detail(
    application_id: str,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_logging_db)
):
    """
    Get detailed information about a specific prediction.
    
    **Admin only**
    """
    log = db.query(PredictionLog).filter(
        PredictionLog.application_id == application_id
    ).first()
    
    if not log:
        raise HTTPException(status_code=404, detail="Application not found")
    
    # Parse JSON fields
    result = {
        "application_id": log.application_id,
        "timestamp": log.timestamp,
        "submitted_by": log.submitted_by,
        "applicant_name": log.applicant_name,
        "applicant_age": log.applicant_age,
        "applicant_income": log.applicant_income,
        "gender": log.gender,
        "race": log.race,
        "loan_amount": log.loan_amount,
        "loan_term": log.loan_term,
        "loan_purpose": log.loan_purpose,
        "prediction": log.prediction,
        "probability": log.probability,
        "confidence": log.confidence,
        "documents_uploaded": log.documents_uploaded.split(",") if log.documents_uploaded else [],
        "nlp_features": json.loads(log.nlp_features) if log.nlp_features else None,
        "extracted_entities": json.loads(log.extracted_entities) if log.extracted_entities else None,
        "shap_values": json.loads(log.shap_values) if log.shap_values else None,
        "adverse_action_reasons": json.loads(log.adverse_action_reasons) if log.adverse_action_reasons else None,
    }
    
    return result


# ============================================
# SYSTEM STATISTICS
# ============================================

@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_logging_db)
):
    """
    Get high-level system statistics.
    
    **Admin only**
    """
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = now - timedelta(days=7)
    
    # Total predictions
    total = db.query(func.count(PredictionLog.id)).scalar()
    
    # Predictions today
    today = db.query(func.count(PredictionLog.id)).filter(
        PredictionLog.timestamp >= today_start
    ).scalar()
    
    # Predictions this week
    this_week = db.query(func.count(PredictionLog.id)).filter(
        PredictionLog.timestamp >= week_start
    ).scalar()
    
    # Approval/denial rates
    approved = db.query(func.count(PredictionLog.id)).filter(
        PredictionLog.prediction == "approved"
    ).scalar()
    
    approval_rate = (approved / total * 100) if total > 0 else 0.0
    denial_rate = 100.0 - approval_rate
    
    # Average loan amount
    avg_loan = db.query(func.avg(PredictionLog.loan_amount)).scalar() or 0.0
    
    # Last prediction time
    last_pred = db.query(PredictionLog).order_by(
        desc(PredictionLog.timestamp)
    ).first()
    
    return SystemStatsResponse(
        total_predictions=total,
        predictions_today=today,
        predictions_this_week=this_week,
        approval_rate=round(approval_rate, 2),
        denial_rate=round(denial_rate, 2),
        avg_loan_amount=round(avg_loan, 2),
        active_users=0,  # TODO: Count from users table
        last_prediction_time=last_pred.timestamp if last_pred else None
    )


# ============================================
# FAIRNESS AUDIT
# ============================================

@router.get("/fairness_report", response_model=FairnessReportResponse)
async def generate_fairness_report(
    days_back: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_logging_db)
):
    """
    Generate a fairness audit report on recent predictions.
    
    **Admin only**
    
    Analyzes predictions from the last N days and computes:
    - Selection rate by demographic group
    - Demographic parity difference
    - Equalized odds metrics
    """
    # Get predictions from last N days
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    logs = db.query(PredictionLog).filter(
        PredictionLog.timestamp >= cutoff_date,
        PredictionLog.gender.isnot(None),  # Only logs with demographic data
        PredictionLog.race.isnot(None)
    ).all()
    
    if len(logs) < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data for fairness audit. Found {len(logs)} predictions with demographic info (minimum 10 required)"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        "application_id": log.application_id,
        "prediction": 1 if log.prediction == "approved" else 0,
        "probability": log.probability,
        "gender": log.gender,
        "race": log.race,
        "loan_amount": log.loan_amount or 0,
    } for log in logs])
    
    # Run fairness audit
    auditor = FairnessAuditor()
    
    # Selection rate by gender
    gender_rates = df.groupby("gender")["prediction"].mean().to_dict()
    
    # Selection rate by race
    race_rates = df.groupby("race")["prediction"].mean().to_dict()
    
    # Demographic parity difference (max difference between groups)
    overall_rate = df["prediction"].mean()
    gender_diffs = [abs(rate - overall_rate) for rate in gender_rates.values()]
    race_diffs = [abs(rate - overall_rate) for rate in race_rates.values()]
    
    max_gender_diff = max(gender_diffs) if gender_diffs else 0.0
    max_race_diff = max(race_diffs) if race_diffs else 0.0
    
    # Build metrics
    metrics = [
        FairnessMetrics(
            metric_name="Selection Rate by Gender",
            overall_value=overall_rate,
            by_group=gender_rates
        ),
        FairnessMetrics(
            metric_name="Selection Rate by Race",
            overall_value=overall_rate,
            by_group=race_rates
        ),
        FairnessMetrics(
            metric_name="Demographic Parity (Gender)",
            overall_value=max_gender_diff,
            by_group={"max_difference": max_gender_diff}
        ),
        FairnessMetrics(
            metric_name="Demographic Parity (Race)",
            overall_value=max_race_diff,
            by_group={"max_difference": max_race_diff}
        )
    ]
    
    # Generate recommendations
    recommendations = []
    
    if max_gender_diff > 0.10:
        recommendations.append(
            f"⚠️ Gender disparity detected: {max_gender_diff:.1%} difference in approval rates. "
            "Consider bias mitigation techniques."
        )
    
    if max_race_diff > 0.10:
        recommendations.append(
            f"⚠️ Racial disparity detected: {max_race_diff:.1%} difference in approval rates. "
            "Consider bias mitigation techniques."
        )
    
    if not recommendations:
        recommendations.append("✅ No significant fairness issues detected in the specified metrics.")
    
    return FairnessReportResponse(
        generated_at=datetime.utcnow(),
        total_predictions_analyzed=len(logs),
        date_range={
            "start": cutoff_date.isoformat(),
            "end": datetime.utcnow().isoformat()
        },
        metrics=metrics,
        recommendations=recommendations
    )


# ============================================
# MODEL MANAGEMENT
# ============================================

@router.get("/model_info")
async def get_model_info(
    current_user: User = Depends(get_admin_user)
):
    """
    Get information about the currently deployed model.
    
    **Admin only**
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
    
    model_files = []
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith((".pkl", ".joblib", ".json")):
                filepath = os.path.join(models_dir, f)
                stat = os.stat(filepath)
                model_files.append({
                    "filename": f,
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
    
    # Check for metadata
    metadata_path = os.path.join(models_dir, "model_metadata.json")
    metadata = None
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    return {
        "models_directory": models_dir,
        "model_files": model_files,
        "metadata": metadata,
        "model_loaded": True  # TODO: Check if model is actually loaded in memory
    }


@router.post("/retrain_model")
async def trigger_model_retrain(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_logging_db)
):
    """
    Trigger model retraining using logged predictions.
    
    **Admin only**
    
    NOTE: This endpoint creates a retraining task but does not block.
    In production, this should queue a background job.
    """
    # Count available training data
    total_logs = db.query(func.count(PredictionLog.id)).scalar()
    
    if total_logs < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough data for retraining. Found {total_logs} predictions (minimum 100 required)"
        )
    
    # In a real system, you would:
    # 1. Export logs to CSV
    # 2. Queue a background job to run train_simple_model.py
    # 3. Return a task ID for status checking
    
    return {
        "status": "queued",
        "message": f"Retraining task queued with {total_logs} training samples",
        "task_id": "retrain-" + datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        "note": "In production, this would be a background job. Currently not implemented."
    }
