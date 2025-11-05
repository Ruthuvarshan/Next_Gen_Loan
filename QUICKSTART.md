# Quick Start Guide

## Next-Generation Loan Origination System

This guide will help you get the system up and running quickly.

## Prerequisites

- Python 3.9 or higher
- Tesseract OCR installed on your system
- Docker (optional, for containerized deployment)
- At least 4GB RAM for model training

## Installation Steps

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd r:\SSF\Next_Gen_Loan

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Install Tesseract OCR

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Install and note the installation path
- Update `.env` file with the path

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### 3. Configure Environment

```bash
# Copy example environment file
copy .env.example .env

# Edit .env and update:
# - TESSERACT_CMD path
# - Database URL (if using PostgreSQL)
# - Other configuration as needed
```

### 4. Create Required Directories

```bash
mkdir -p data\raw data\processed data\sample data\uploads
mkdir -p models logs docs\images
mkdir -p notebooks tests scripts
```

## Running the System

### Option 1: Development Mode (Local)

```bash
# Start the API server
python -m src.api.main

# Or use uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Option 2: Production Mode (Docker)

```bash
# Build and start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Testing the API

### Using curl (Windows CMD):

**Health Check:**
```cmd
curl http://localhost:8000/health
```

**Prediction (with JSON data only):**
```cmd
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"application_data\": {\"credit_score\": 720, \"age\": 35, \"loan_amount\": 15000, \"loan_term\": 36}}"
```

**Explanation:**
```cmd
curl -X POST "http://localhost:8000/explain" ^
  -H "Content-Type: application/json" ^
  -d "{\"applicant_id\": \"APP-20241104103000\"}"
```

### Using Python requests:

```python
import requests

# Prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "application_data": {
            "credit_score": 720,
            "age": 35,
            "loan_amount": 15000,
            "loan_term": 36,
            "annual_income": 60000
        }
    }
)
print(response.json())
```

## Training a Model

**Note:** You'll need training data first. The system expects a CSV file with loan application data.

```bash
# Basic training
python scripts/train_model.py --data data/processed/training_data.csv --output models/

# Training with fairness mitigation
python scripts/train_model.py ^
  --data data/processed/training_data.csv ^
  --output models/ ^
  --fairness ^
  --constraint equalized_odds ^
  --hyperparameter-tuning
```

## Project Structure Overview

```
Next_Gen_Loan/
├── src/                          # Source code
│   ├── modules/                  # Core modules
│   │   ├── idp_engine.py        # Intelligent Document Processing
│   │   ├── nlp_features.py      # NLP Feature Engineering
│   │   ├── risk_model.py        # XGBoost Risk Model
│   │   ├── xai_explainer.py     # SHAP Explainability
│   │   └── fairness_audit.py    # Fairness Auditing
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # Main API app
│   │   └── schemas.py           # Request/Response schemas
│   └── utils/                    # Utilities
│       ├── config.py            # Configuration
│       └── preprocessing.py     # Data preprocessing
├── models/                       # Trained models (gitignored)
├── data/                         # Data directories (gitignored)
├── scripts/                      # Training/utility scripts
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── docs/                         # Documentation and images
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker image definition
├── docker-compose.yml           # Docker Compose configuration
└── README.md                     # Main documentation
```

## Common Issues and Solutions

### Issue: "Tesseract not found"
**Solution:** 
- Ensure Tesseract is installed
- Update `TESSERACT_CMD` in `.env` file with correct path
- Windows example: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`

### Issue: "spaCy model not found"
**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: "Model file not found"
**Solution:**
- The system needs a trained model in `models/xgboost_model.pkl`
- Either train a model using `scripts/train_model.py`
- Or place a pre-trained model in the `models/` directory

### Issue: Port 8000 already in use
**Solution:**
```bash
# Use a different port
uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# Or update API_PORT in .env file
```

## Next Steps

1. **Prepare Training Data**: Create or obtain loan application training data
2. **Train Model**: Run the training script to create your risk model
3. **Test Endpoints**: Use the interactive docs at `/docs` to test
4. **Deploy**: Use Docker for production deployment
5. **Monitor**: Set up logging and fairness monitoring

## Getting Help

- Check the main README.md for detailed documentation
- Review the interactive API docs at `/docs`
- Check logs in the `logs/` directory
- Review code comments in individual modules

## Security Reminders

Before deploying to production:
- [ ] Change SECRET_KEY in .env
- [ ] Configure CORS properly in main.py
- [ ] Set up proper authentication/authorization
- [ ] Use HTTPS
- [ ] Review and restrict API rate limits
- [ ] Set up database backups
- [ ] Configure proper logging and monitoring
