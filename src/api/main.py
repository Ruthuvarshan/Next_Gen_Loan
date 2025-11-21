"""
FastAPI Main Application
Production-ready loan origination microservice with IDP, NLP, XAI, and Fairness
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
import joblib
from pathlib import Path
from typing import Optional, Dict, List
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

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
from src.utils.database import init_databases, get_users_db, get_logging_db, User, PredictionLog
from src.utils.auth import authenticate_user, create_access_token, get_current_user
from src.api.admin import router as admin_router
from sqlalchemy.orm import Session

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# PySpark imports (optional - fallback to pandas if not available)
PYSPARK_AVAILABLE = False
try:
    from src.utils.spark_config import get_or_create_spark, stop_spark_session
    from src.modules.spark_nlp_features import SparkNLPFeatureEngine, extract_nlp_features_from_text
    from src.modules.spark_risk_model import SparkRiskScorer
    PYSPARK_AVAILABLE = True
    logger.info("PySpark modules loaded successfully")
except ImportError as e:
    logger.warning(f"PySpark not available, falling back to pandas: {e}")
    PYSPARK_AVAILABLE = False

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
    allow_origins=[
        "http://localhost:3000",  # User Portal
        "http://localhost:3001",  # Admin Dashboard
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include admin router
app.include_router(admin_router, prefix="/api")

# Global state for models (loaded on startup)
class ModelState:
    """Global state for loaded models and processors."""
    risk_model: Optional[CreditRiskModel] = None
    xai_explainer: Optional[XAIExplainer] = None
    idp_engine: Optional[IDPEngine] = None
    nlp_engine: Optional[NLPFeatureEngine] = None
    preprocessor: Optional[DataPreprocessor] = None
    prediction_cache: Dict = {}  # Cache recent predictions for explain endpoint
    
    # PySpark models (if available)
    use_pyspark: bool = False
    spark_risk_scorer: Optional[SparkRiskScorer] = None
    spark_nlp_engine: Optional[SparkNLPFeatureEngine] = None

state = ModelState()


@app.on_event("startup")
async def startup_event():
    """Load models and initialize processors on startup."""
    logger.info("Starting up application...")
    
    try:
        # Initialize databases
        init_databases()
        
        # Initialize IDP Engine
        logger.info("Initializing IDP Engine...")
        state.idp_engine = IDPEngine()
        
        # Check if PySpark models are available
        spark_model_path = Path("models/spark_xgboost_model")
        
        if PYSPARK_AVAILABLE and spark_model_path.exists():
            logger.info("PySpark models detected. Loading PySpark pipeline...")
            state.use_pyspark = True
            
            # Initialize Spark session
            spark = get_or_create_spark()
            
            # Initialize Spark NLP engine
            logger.info("Initializing Spark NLP Feature Engine...")
            state.spark_nlp_engine = SparkNLPFeatureEngine(spark)
            
            # Load Spark risk scorer
            logger.info(f"Loading Spark XGBoost model from {spark_model_path}...")
            # Load feature columns from metadata
            metadata_path = Path("models/model_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    feature_cols = metadata.get('feature_columns', [])
            else:
                feature_cols = []  # Will be inferred
            
            state.spark_risk_scorer = SparkRiskScorer(
                model_path=str(spark_model_path),
                feature_cols=feature_cols
            )
            
            logger.info("PySpark models loaded successfully")
            
        else:
            # Fall back to pandas-based models
            logger.info("Using pandas-based models...")
            state.use_pyspark = False
            
            # Initialize NLP Feature Engine (pandas)
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


@app.post("/api/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_users_db)
):
    """
    OAuth2 compatible token endpoint.
    
    Returns a JWT access token upon successful authentication.
    
    **Body:**
    - username: User's username
    - password: User's password
    
    **Returns:**
    - access_token: JWT token
    - token_type: "bearer"
    - user_info: Basic user information (username, role, full_name)
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create JWT token
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_info": {
            "username": user.username,
            "role": user.role,
            "full_name": user.full_name,
            "email": user.email
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    # Multipart form data fields
    applicant_name: str = Form(...),
    credit_score: int = Form(...),
    age: int = Form(...),
    loan_amount: float = Form(...),
    loan_term: int = Form(...),
    annual_income: Optional[float] = Form(None),
    loan_purpose: Optional[str] = Form(None),
    sex: Optional[str] = Form(None),
    race: Optional[str] = Form(None),
    age_group: Optional[str] = Form(None),
    zip_code: Optional[str] = Form(None),
    bank_statement_text: Optional[str] = Form(None),
    
    # File uploads
    paystub: Optional[UploadFile] = File(None),
    bank_statement: Optional[UploadFile] = File(None),
    
    # Authentication
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_logging_db)
):
    """
    Complete prediction pipeline with multipart/form-data support:
    1. Process uploaded documents (IDP)
    2. Extract NLP features from bank statement
    3. Assemble feature matrix
    4. Generate prediction
    5. Log to database for fairness monitoring
    
    **Requires authentication** - Include JWT token in Authorization header.
    """
    try:
        # Generate unique application ID
        applicant_id = f"APP-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        
        logger.info(f"Processing prediction request {applicant_id} for {applicant_name} by user {current_user.username}")
        
        # Check if model is loaded
        if state.risk_model is None or state.risk_model.model is None:
            raise HTTPException(status_code=503, detail="Risk model not loaded")
        
        # ============ Phase 1: Traditional Features ============
        traditional_features = {
            'credit_score': credit_score,
            'age': age,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'annual_income': annual_income or 0
        }
        
        # ============ Phase 2: IDP Processing ============
        idp_features = {}
        extracted_entities = {}
        documents_uploaded = []
        
        if paystub:
            logger.info("Processing paystub document...")
            documents_uploaded.append(paystub.filename)
            
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
            
            extracted_entities.update(idp_result.get('entities', {}))
            
            # Clean up temp file
            Path(tmp_path).unlink()
        
        # ============ Phase 3: NLP Feature Engineering ============
        nlp_features = {}
        bank_text = bank_statement_text
        
        if bank_statement:
            logger.info("Processing bank statement document...")
            documents_uploaded.append(bank_statement.filename)
            
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
                loan_purpose=loan_purpose
            )
            # Remove non-numeric features
            nlp_features = {k: v for k, v in nlp_features.items() if isinstance(v, (int, float))}
        
        # ============ Phase 4: Feature Assembly ============
        all_features = {**traditional_features, **idp_features, **nlp_features}
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([all_features])
        
        # For sklearn pipeline models, we need to provide all expected columns
        # The pipeline will handle preprocessing internally
        if hasattr(state.risk_model.model, 'named_steps'):
            # Pipeline model - fill in missing columns with defaults
            expected_features = {
                'application_id': applicant_id,
                'applicant_name': applicant_name,
                'age': age,
                'annual_income': annual_income or 0,
                'employment_length': 5.0,  # Default 5 years
                'sex': sex or 'Unknown',
                'race': race or 'Unknown',
                'credit_score': credit_score,
                'debt_to_income_ratio': 0.3,  # Will be calculated if income available
                'num_credit_lines': 5,
                'num_derogatory_marks': 0,
                'months_since_last_delinquency': 24.0,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'loan_purpose': loan_purpose or 'other',
                'interest_rate': 0.10,  # 10% default
                'avg_monthly_balance': (annual_income or 0) / 12 if annual_income else 0,
                'num_overdrafts': 0,
                'num_late_fees': 0,
                'monthly_income_deposits': (annual_income or 0) / 12 if annual_income else 0
            }
            
            # Override with NLP features if available
            expected_features.update(nlp_features)
            # Override with IDP features if available
            expected_features.update(idp_features)
            # Override with explicitly provided features
            expected_features.update(traditional_features)
            
            # Calculate DTI if we have income
            if annual_income and annual_income > 0:
                expected_features['debt_to_income_ratio'] = (loan_amount / loan_term) / (annual_income / 12)
            
            feature_df = pd.DataFrame([expected_features])
        else:
            # Old model format - use existing logic
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
            decision = "approved"
            confidence_label = "High" if probability < 0.3 else "Medium"
            confidence_score = 0.9 if probability < 0.3 else 0.7
        else:
            decision = "denied"
            confidence_label = "High" if probability > settings.high_risk_threshold else "Medium"
            confidence_score = 0.9 if probability > settings.high_risk_threshold else 0.7
        
        # ============ Phase 6: Generate SHAP Explanation (for denied applications) ============
        shap_values_json = None
        adverse_action_reasons = None
        
        if decision == "denied":
            try:
                # Initialize explainer if needed
                if state.xai_explainer.explainer is None:
                    logger.info("Initializing SHAP explainer...")
                    state.xai_explainer.initialize_explainer(feature_df)
                
                # Calculate SHAP values
                shap_values = state.xai_explainer.calculate_shap_values(feature_df)
                instance_shap = shap_values[0]
                feature_names = feature_df.columns.tolist()
                
                # Generate adverse action reasons
                adverse_action_reasons = get_simple_adverse_action_reasons(
                    instance_shap,
                    feature_names,
                    top_n=settings.shap_top_n_reasons
                )
                
                # Serialize SHAP values for database
                shap_values_json = json.dumps({
                    "feature_names": feature_names,
                    "shap_values": instance_shap.tolist(),
                    "base_value": float(state.xai_explainer.explainer.expected_value)
                })
                
            except Exception as e:
                logger.warning(f"Could not generate SHAP explanation: {str(e)}")
        
        # ============ Phase 7: Database Logging ============
        log_entry = PredictionLog(
            application_id=applicant_id,
            timestamp=datetime.utcnow(),
            submitted_by=current_user.username,
            applicant_name=applicant_name,
            applicant_age=age,
            applicant_income=annual_income,
            gender=sex,
            race=race,
            loan_amount=loan_amount,
            loan_term=loan_term,
            loan_purpose=loan_purpose,
            prediction=decision,
            probability=probability,
            confidence=confidence_score,
            nlp_features=json.dumps(nlp_features) if nlp_features else None,
            extracted_entities=json.dumps(extracted_entities) if extracted_entities else None,
            shap_values=shap_values_json,
            adverse_action_reasons=json.dumps(adverse_action_reasons) if adverse_action_reasons else None,
            documents_uploaded=",".join(documents_uploaded) if documents_uploaded else None
        )
        
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        logger.info(f"Prediction logged to database: ID {log_entry.id}")
        
        # ============ Phase 8: Cache for Explanation Endpoint ============
        state.prediction_cache[applicant_id] = {
            'features': feature_df,
            'decision': decision,
            'probability': probability,
            'timestamp': datetime.now()
        }
        
        logger.info(f"Prediction complete: {decision} (prob={probability:.4f})")
        
        return PredictionResponse(
            applicant_id=applicant_id,
            decision=decision.capitalize(),  # Return as "Approved" or "Denied"
            probability=probability,
            confidence=confidence_label,
            adverse_action_reasons=adverse_action_reasons if decision == "denied" else None
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
