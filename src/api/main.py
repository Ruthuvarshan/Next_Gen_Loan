"""
FastAPI Main Application
Production-ready loan origination microservice with IDP, NLP, XAI, and Fairness
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import joblib
from pathlib import Path
from typing import Optional, Dict, List
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ExplanationRequest,
    ExplanationResponse,
    HealthResponse,
    ErrorResponse,
    FeatureContribution
)
from src.modules.idp_engine import IDPEngine
from src.modules.nlp_features import NLPFeatureEngine
from src.modules.risk_model import CreditRiskModel
from src.modules.xai_explainer import XAIExplainer, get_simple_adverse_action_reasons
from src.utils.config import settings
from src.utils.preprocessing import DataPreprocessor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Next-Generation Loan Origination System with IDP, NLP, XAI, and Fairness Auditing",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models (loaded on startup)
class ModelState:
    """Global state for loaded models and processors."""
    risk_model: Optional[CreditRiskModel] = None
    xai_explainer: Optional[XAIExplainer] = None
    idp_engine: Optional[IDPEngine] = None
    nlp_engine: Optional[NLPFeatureEngine] = None
    preprocessor: Optional[DataPreprocessor] = None
    prediction_cache: Dict = {}  # Cache recent predictions for explain endpoint

state = ModelState()


@app.on_event("startup")
async def startup_event():
    """Load models and initialize processors on startup."""
    logger.info("Starting up application...")
    
    try:
        # Initialize IDP Engine
        logger.info("Initializing IDP Engine...")
        state.idp_engine = IDPEngine()
        
        # Initialize NLP Feature Engine
        logger.info("Initializing NLP Feature Engine...")
        state.nlp_engine = NLPFeatureEngine()
        
        # Initialize preprocessor
        state.preprocessor = DataPreprocessor()
        
        # Load trained risk model
        model_path = settings.get_model_path()
        if model_path.exists():
            logger.info(f"Loading risk model from {model_path}...")
            state.risk_model = CreditRiskModel(model_path=model_path)
            logger.info("Risk model loaded successfully")
        else:
            logger.warning(f"Risk model not found at {model_path}. Prediction endpoint will be unavailable.")
        
        # Initialize XAI explainer (will be fully initialized with background data on first use)
        if state.risk_model and state.risk_model.model:
            logger.info("Initializing XAI Explainer...")
            state.xai_explainer = XAIExplainer(state.risk_model.model)
            logger.info("XAI Explainer initialized")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict": "/predict (POST)",
            "explain": "/explain (POST)"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        model_loaded=state.risk_model is not None and state.risk_model.model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    paystub: Optional[UploadFile] = File(None),
    bank_statement: Optional[UploadFile] = File(None)
):
    """
    Complete prediction pipeline:
    1. Process uploaded documents (IDP)
    2. Extract NLP features from bank statement
    3. Assemble feature matrix
    4. Generate prediction
    5. Log for fairness monitoring
    """
    try:
        logger.info(f"Processing prediction request for applicant: {request.applicant_id}")
        
        # Check if model is loaded
        if state.risk_model is None or state.risk_model.model is None:
            raise HTTPException(status_code=503, detail="Risk model not loaded")
        
        # Generate applicant ID if not provided
        applicant_id = request.applicant_id or f"APP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # ============ Phase 1: Traditional Features ============
        traditional_features = {
            'credit_score': request.application_data.credit_score,
            'age': request.application_data.age,
            'loan_amount': request.application_data.loan_amount,
            'loan_term': request.application_data.loan_term,
            'annual_income': request.application_data.annual_income or 0
        }
        
        # ============ Phase 2: IDP Processing ============
        idp_features = {}
        
        if paystub:
            logger.info("Processing paystub document...")
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                content = await paystub.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Process with IDP
            idp_result = state.idp_engine.process_document(tmp_path, document_type='paystub')
            
            if 'net_income' in idp_result:
                idp_features['verified_net_income'] = idp_result['net_income']
            if 'gross_income' in idp_result:
                idp_features['verified_gross_income'] = idp_result['gross_income']
            
            # Clean up temp file
            Path(tmp_path).unlink()
        
        # ============ Phase 3: NLP Feature Engineering ============
        nlp_features = {}
        
        bank_text = request.bank_statement_text
        
        if bank_statement:
            logger.info("Processing bank statement document...")
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                content = await bank_statement.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Extract text with IDP
            statement_result = state.idp_engine.process_document(tmp_path, document_type='bank_statement')
            bank_text = statement_result.get('raw_text', '')
            
            # Clean up temp file
            Path(tmp_path).unlink()
        
        if bank_text:
            logger.info("Extracting NLP features from bank statement...")
            nlp_features = state.nlp_engine.extract_features(
                bank_text,
                loan_purpose=request.application_data.loan_purpose
            )
            # Remove non-numeric features
            nlp_features = {k: v for k, v in nlp_features.items() if isinstance(v, (int, float))}
        
        # ============ Phase 4: Feature Assembly ============
        all_features = {**traditional_features, **idp_features, **nlp_features}
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        # Handle missing values
        feature_df = state.preprocessor.handle_missing_values(feature_df)
        
        # Ensure all expected features are present (fill with 0 if missing)
        if state.risk_model.feature_names:
            for feature in state.risk_model.feature_names:
                if feature not in feature_df.columns:
                    feature_df[feature] = 0
            
            # Reorder columns to match training
            feature_df = feature_df[state.risk_model.feature_names]
        
        # ============ Phase 5: Prediction ============
        logger.info("Generating prediction...")
        predictions, probabilities = state.risk_model.predict(feature_df)
        
        prediction = predictions[0]
        probability = float(probabilities[0])
        
        # Determine decision based on threshold and probability
        if probability < settings.approval_threshold:
            decision = "Approve"
            confidence = "High" if probability < 0.3 else "Medium"
        else:
            decision = "Deny"
            confidence = "High" if probability > settings.high_risk_threshold else "Medium"
        
        # ============ Phase 6: Cache for Explanation ============
        state.prediction_cache[applicant_id] = {
            'features': feature_df,
            'decision': decision,
            'probability': probability,
            'timestamp': datetime.now()
        }
        
        # ============ Phase 7: Fairness Logging ============
        if settings.enable_fairness_logging:
            sensitive_attrs = {
                'sex': request.application_data.sex,
                'age_group': request.application_data.age_group,
                'zip_code': request.application_data.zip_code
            }
            # In production, this would log to database
            logger.info(f"Fairness log: {applicant_id}, decision={decision}, sensitive_attrs={sensitive_attrs}")
        
        logger.info(f"Prediction complete: {decision} (prob={probability:.4f})")
        
        return PredictionResponse(
            applicant_id=applicant_id,
            decision=decision,
            probability=probability,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain", response_model=ExplanationResponse)
async def explain(request: ExplanationRequest):
    """
    Generate explanation for a previous prediction using SHAP.
    Provides adverse action reasons for denied applications (ECOA compliance).
    """
    try:
        logger.info(f"Generating explanation for applicant: {request.applicant_id}")
        
        # Check if prediction exists in cache
        if request.applicant_id not in state.prediction_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Prediction not found for applicant {request.applicant_id}. Call /predict first."
            )
        
        # Retrieve cached prediction
        cached = state.prediction_cache[request.applicant_id]
        feature_df = cached['features']
        decision = cached['decision']
        probability = cached['probability']
        
        # Initialize explainer if not already done
        if state.xai_explainer.explainer is None:
            logger.info("Initializing SHAP explainer with background data...")
            # Use the cached features as background (in production, use training set sample)
            state.xai_explainer.initialize_explainer(feature_df)
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        shap_values = state.xai_explainer.calculate_shap_values(feature_df)
        
        # Get feature contributions
        instance_shap = shap_values[0]
        feature_names = feature_df.columns.tolist()
        feature_values = feature_df.iloc[0]
        
        contributions = pd.DataFrame({
            'feature': feature_names,
            'value': feature_values.values,
            'shap_value': instance_shap
        }).sort_values('shap_value', ascending=True)
        
        top_negative = contributions.head(5).to_dict('records')
        top_positive = contributions.tail(5).to_dict('records')
        
        # Generate adverse action reasons if denied
        adverse_action_reasons = None
        if decision == "Deny":
            logger.info("Generating adverse action reasons...")
            adverse_action_reasons = get_simple_adverse_action_reasons(
                instance_shap,
                feature_names,
                top_n=settings.shap_top_n_reasons
            )
        
        logger.info("Explanation generated successfully")
        
        return ExplanationResponse(
            applicant_id=request.applicant_id,
            decision=decision,
            probability=probability,
            base_value=float(state.xai_explainer.explainer.expected_value),
            top_positive_features=[
                FeatureContribution(**item) for item in top_positive
            ],
            top_negative_features=[
                FeatureContribution(**item) for item in top_negative
            ],
            adverse_action_reasons=adverse_action_reasons
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
