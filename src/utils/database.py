"""
SQLite database models and session management.
Creates two databases: users.db (authentication) and logging.db (predictions).
"""

import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Base directory for databases
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(DB_DIR, exist_ok=True)

# Database URLs
USERS_DB_URL = f"sqlite:///{os.path.join(DB_DIR, 'users.db')}"
LOGGING_DB_URL = f"sqlite:///{os.path.join(DB_DIR, 'logging.db')}"

# Create engines
users_engine = create_engine(USERS_DB_URL, connect_args={"check_same_thread": False})
logging_engine = create_engine(LOGGING_DB_URL, connect_args={"check_same_thread": False})

# Session factories
UsersSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=users_engine)
LoggingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=logging_engine)

# Base classes
UsersBase = declarative_base()
LoggingBase = declarative_base()


# ============================================
# USERS DATABASE MODELS
# ============================================

class User(UsersBase):
    """User account model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(String(20), nullable=False, default="user")  # "user" or "admin"
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


# ============================================
# LOGGING DATABASE MODELS
# ============================================

class PredictionLog(LoggingBase):
    """Log of every prediction made by the system."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # User who submitted
    submitted_by = Column(String(100), nullable=True)  # Username of loan officer
    
    # Applicant info
    applicant_name = Column(String(255), nullable=True)
    applicant_age = Column(Integer, nullable=True)
    applicant_income = Column(Float, nullable=True)
    
    # Sensitive attributes (for fairness auditing)
    gender = Column(String(20), nullable=True)
    race = Column(String(50), nullable=True)
    
    # Loan details
    loan_amount = Column(Float, nullable=True)
    loan_term = Column(Integer, nullable=True)
    loan_purpose = Column(String(100), nullable=True)
    
    # Model prediction
    prediction = Column(String(20), nullable=False)  # "approved" or "denied"
    probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    
    # Features (JSON stored as text)
    nlp_features = Column(Text, nullable=True)  # JSON string
    extracted_entities = Column(Text, nullable=True)  # JSON string
    
    # Explainability
    shap_values = Column(Text, nullable=True)  # JSON string
    adverse_action_reasons = Column(Text, nullable=True)  # JSON array
    
    # Documents uploaded
    documents_uploaded = Column(Text, nullable=True)  # Comma-separated filenames


# ============================================
# DATABASE INITIALIZATION
# ============================================

def init_databases():
    """Create all tables in both databases."""
    print("🗄️  Initializing databases...")
    
    # Create users database tables
    UsersBase.metadata.create_all(bind=users_engine)
    print(f"   ✓ Users database: {os.path.join(DB_DIR, 'users.db')}")
    
    # Create logging database tables
    LoggingBase.metadata.create_all(bind=logging_engine)
    print(f"   ✓ Logging database: {os.path.join(DB_DIR, 'logging.db')}")


# ============================================
# DEPENDENCY FUNCTIONS
# ============================================

def get_users_db() -> Session:
    """Dependency to get users database session."""
    db = UsersSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_logging_db() -> Session:
    """Dependency to get logging database session."""
    db = LoggingSessionLocal()
    try:
        yield db
    finally:
        db.close()
