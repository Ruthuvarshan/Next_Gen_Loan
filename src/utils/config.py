"""
Configuration management for the Loan Origination System.
Loads environment variables and provides centralized configuration access.
"""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = Field(default="Next Gen Loan Origination System", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=4, env="API_WORKERS")
    
    # Paths
    base_dir: Path = Path(__file__).parent.parent.parent
    model_path: str = Field(default="models/xgboost_model.pkl", env="MODEL_PATH")
    scaler_path: str = Field(default="models/scaler.pkl", env="SCALER_PATH")
    spacy_ner_path: str = Field(default="models/spacy_ner", env="SPACY_NER_PATH")
    spacy_model: str = Field(default="en_core_web_sm", env="SPACY_MODEL")
    
    # Prediction
    approval_threshold: float = Field(default=0.50, env="APPROVAL_THRESHOLD")
    high_risk_threshold: float = Field(default=0.70, env="HIGH_RISK_THRESHOLD")
    
    # IDP
    tesseract_cmd: str = Field(default="/usr/bin/tesseract", env="TESSERACT_CMD")
    max_upload_size_mb: int = Field(default=10, env="MAX_UPLOAD_SIZE_MB")
    allowed_file_types: str = Field(default="pdf,jpg,jpeg,png", env="ALLOWED_FILE_TYPES")
    
    # NLP
    transaction_lookback_months: int = Field(default=6, env="TRANSACTION_LOOKBACK_MONTHS")
    min_transactions_required: int = Field(default=10, env="MIN_TRANSACTIONS_REQUIRED")
    
    # Fairness
    enable_fairness_logging: bool = Field(default=True, env="ENABLE_FAIRNESS_LOGGING")
    sensitive_attributes: str = Field(default="sex,age_group,race_proxy", env="SENSITIVE_ATTRIBUTES")
    fairness_threshold: float = Field(default=0.05, env="FAIRNESS_THRESHOLD")
    
    # Database
    database_url: str = Field(
        default="postgresql://loanuser:loanpass@localhost:5432/loan_audit",
        env="DATABASE_URL"
    )
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="logs/application.log", env="LOG_FILE")
    
    # Security
    secret_key: str = Field(default="change-me-in-production", env="SECRET_KEY")
    api_key_header: str = Field(default="X-API-Key", env="API_KEY_HEADER")
    
    # SHAP
    shap_sample_size: int = Field(default=100, env="SHAP_SAMPLE_SIZE")
    shap_top_n_reasons: int = Field(default=3, env="SHAP_TOP_N_REASONS")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_period: int = Field(default=3600, env="RATE_LIMIT_PERIOD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def allowed_file_types_list(self) -> List[str]:
        """Return allowed file types as a list."""
        return [ft.strip() for ft in self.allowed_file_types.split(",")]
    
    @property
    def sensitive_attributes_list(self) -> List[str]:
        """Return sensitive attributes as a list."""
        return [attr.strip() for attr in self.sensitive_attributes.split(",")]
    
    def get_model_path(self) -> Path:
        """Get absolute path to the model file."""
        return self.base_dir / self.model_path
    
    def get_scaler_path(self) -> Path:
        """Get absolute path to the scaler file."""
        return self.base_dir / self.scaler_path
    
    def get_spacy_ner_path(self) -> Path:
        """Get absolute path to the spaCy NER model."""
        return self.base_dir / self.spacy_ner_path


# Global settings instance
settings = Settings()
