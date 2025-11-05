# Two-Window Architecture - Complete Setup Guide

This project implements a **two-window architecture** with JWT authentication, database logging, and complete separation of concerns:

1. **User Portal** (Port 3000) - Loan Officer Interface
2. **Admin Dashboard** (Port 3001) - Risk Management Interface  
3. **Backend API** (Port 8000) - FastAPI with SQLite databases

---

## 🗄️ Database Architecture

The system uses **SQLite** (server-less databases):

- **`data/users.db`** - User authentication (usernames, hashed passwords, roles)
- **`data/logging.db`** - Prediction logs (every application submitted with full details)

**No XAMPP or MySQL needed!** Python manages everything automatically.

---

## 🚀 Quick Start

### 1. Backend Setup

```cmd
# Install Python dependencies (including auth libraries)
pip install -r requirements.txt

# Seed the database with demo users
python scripts\seed_users.py

# Start the FastAPI backend
uvicorn src.api.main:app --reload --port 8000
```

**Demo Credentials:**
- **Loan Officers**: `loan_officer` / `officer123`
- **Admins**: `admin` / `admin123`

### 2. User Portal Setup

```cmd
cd user-portal
npm install
npm run dev
```

Access at: **http://localhost:3000**

### 3. Admin Dashboard Setup

```cmd
cd admin-dashboard
npm install
npm run dev
```

Access at: **http://localhost:3001**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      USERS                                   │
│  👨‍💼 Loan Officers          👑 Risk Analysts/Admins        │
└───────────┬────────────────────────────────┬────────────────┘
            │                                │
            ▼                                ▼
    ┌───────────────┐              ┌───────────────┐
    │ USER PORTAL   │              │ ADMIN DASH    │
    │ Port 3000     │              │ Port 3001     │
    │ React + MUI   │              │ React + AntD  │
    └───────┬───────┘              └───────┬───────┘
            │                              │
            │         ┌────────────────────┤
            │         │                    │
            ▼         ▼                    ▼
        ┌─────────────────────────────────────────┐
        │      FASTAPI BACKEND (Port 8000)        │
        │  ┌────────────┐   ┌────────────────┐   │
        │  │ /api/token │   │ /predict       │   │
        │  │ (login)    │   │ (with JWT)     │   │
        │  └────────────┘   └────────────────┘   │
        │  ┌────────────────────────────────────┐ │
        │  │  ADMIN ENDPOINTS (/api/admin/...)  │ │
        │  │  - /prediction_log                 │ │
        │  │  - /fairness_report                │ │
        │  │  - /stats                          │ │
        │  │  - /model_info                     │ │
        │  └────────────────────────────────────┘ │
        └───────────┬───────────────┬─────────────┘
                    │               │
                    ▼               ▼
            ┌──────────────┐  ┌──────────────┐
            │  users.db    │  │ logging.db   │
            │  (Auth)      │  │ (Predictions)│
            └──────────────┘  └──────────────┘
```

---

## 🔐 Authentication Flow

1. **User submits** username + password to `POST /api/token`
2. **Backend verifies** credentials against `users.db`
3. **JWT token returned** with user role embedded
4. **Frontend stores** token in localStorage
5. **Every API request** includes `Authorization: Bearer <token>` header
6. **Backend validates** token and checks role for admin endpoints

---

## 📋 User Portal Features

**Login Page** (`/login`)
- Username/password form
- JWT token retrieval
- Automatic redirect to dashboard

**Dashboard** (`/`)
- Welcome message with user's name
- "New Application" card
- System features overview
- Logout button

**New Application Wizard** (`/application/new`)
- **Step 1**: Applicant details (name, credit score, age, demographics)
- **Step 2**: Loan details (amount, term, purpose)
- **Step 3**: Document uploads (paystub PDF, bank statement PDF)
- **Step 4**: Review & Submit
- **Real-time processing** feedback during submission

**Results Page** (`/result/:id`)
- ✅ **Approved**: Shows probability, confidence, celebration UI
- ❌ **Denied**: Shows adverse action reasons (ECOA compliant)
- Links to dashboard and new application

---

## 👑 Admin Dashboard Features

_(To be implemented in next phase)_

**Login Page** - Admin credentials only  
**System Stats** - Total predictions, approval rates  
**Prediction Log Table** - Searchable, filterable history  
**Fairness Audit** - Charts showing demographic parity  
**Model Management** - Retrain trigger, model info

---

## 🛠️ Backend Enhancements

### New Files Created:

1. **`src/utils/database.py`**
   - SQLAlchemy models for `User` and `PredictionLog`
   - Database initialization functions
   - Session management

2. **`src/utils/auth.py`**
   - Password hashing with bcrypt
   - JWT token creation/verification
   - OAuth2 dependencies (`get_current_user`, `get_admin_user`)

3. **`src/api/admin.py`**
   - `GET /admin/prediction_log` - Paginated prediction history
   - `GET /admin/prediction_log/{id}` - Detailed prediction view
   - `GET /admin/stats` - System statistics
   - `GET /admin/fairness_report` - Bias detection report
   - `GET /admin/model_info` - Model file information
   - `POST /admin/retrain_model` - Trigger retraining

4. **`scripts/seed_users.py`**
   - Populates `users.db` with demo accounts

### Modified Files:

- **`src/api/main.py`**
  - Added `POST /api/token` endpoint for login
  - Modified `/predict` to accept **multipart/form-data**
  - Added JWT authentication requirement
  - Logs every prediction to `logging.db`
  - Generates SHAP explanations for denied applications
  - Returns adverse action reasons in response

- **`requirements.txt`**
  - Added `python-jose[cryptography]` for JWT
  - Added `passlib[bcrypt]` for password hashing

---

## 📊 Database Schemas

### Users Table (`users.db`)

| Column          | Type     | Description                |
|-----------------|----------|----------------------------|
| id              | Integer  | Primary key                |
| username        | String   | Unique username            |
| email           | String   | Email address              |
| hashed_password | String   | Bcrypt hashed password     |
| full_name       | String   | Display name               |
| role            | String   | "user" or "admin"          |
| is_active       | Boolean  | Account active status      |
| created_at      | DateTime | Registration timestamp     |
| last_login      | DateTime | Last login timestamp       |

### Predictions Table (`logging.db`)

| Column                  | Type     | Description                        |
|-------------------------|----------|------------------------------------|
| id                      | Integer  | Primary key                        |
| application_id          | String   | Unique app ID (APP-YYYYMMDD...)    |
| timestamp               | DateTime | Submission time                    |
| submitted_by            | String   | Username of loan officer           |
| applicant_name          | String   | Applicant's name                   |
| applicant_age           | Integer  | Age                                |
| applicant_income        | Float    | Annual income                      |
| gender                  | String   | For fairness auditing              |
| race                    | String   | For fairness auditing              |
| loan_amount             | Float    | Requested amount                   |
| loan_term               | Integer  | Term in months                     |
| loan_purpose            | String   | Purpose of loan                    |
| prediction              | String   | "approved" or "denied"             |
| probability             | Float    | Risk probability (0-1)             |
| confidence              | Float    | Confidence score                   |
| nlp_features            | Text     | JSON of NLP features               |
| extracted_entities      | Text     | JSON of IDP entities               |
| shap_values             | Text     | JSON of SHAP explanations          |
| adverse_action_reasons  | Text     | JSON array of denial reasons       |
| documents_uploaded      | Text     | Comma-separated filenames          |

---

## 🧪 Testing the System

### 1. Seed Demo Users
```cmd
python scripts\seed_users.py
```

### 2. Start Backend
```cmd
uvicorn src.api.main:app --reload --port 8000
```

### 3. Test Login Endpoint
```cmd
curl -X POST "http://localhost:8000/api/token" ^
  -H "Content-Type: application/x-www-form-urlencoded" ^
  -d "username=loan_officer&password=officer123"
```

Expected response:
```json
{
  "access_token": "eyJ...",
  "token_type": "bearer",
  "user_info": {
    "username": "loan_officer",
    "role": "user",
    "full_name": "Jane Smith",
    "email": "officer@loansystem.com"
  }
}
```

### 4. Test Prediction with Token
```cmd
curl -X POST "http://localhost:8000/predict" ^
  -H "Authorization: Bearer YOUR_TOKEN_HERE" ^
  -F "applicant_name=John Doe" ^
  -F "credit_score=720" ^
  -F "age=35" ^
  -F "loan_amount=15000" ^
  -F "loan_term=36"
```

### 5. Test Admin Endpoint
```cmd
curl -X GET "http://localhost:8000/api/admin/stats" ^
  -H "Authorization: Bearer ADMIN_TOKEN_HERE"
```

---

## 🎯 Next Steps

1. ✅ **Backend complete** - Authentication, logging, admin endpoints
2. ✅ **User Portal complete** - Login, wizard, results
3. ⏳ **Admin Dashboard** - Create with Ant Design + Recharts
4. ⏳ **Docker Compose** - Orchestrate all 3 services
5. ⏳ **Production** - Environment variables, HTTPS, rate limiting

---

## 📝 Environment Variables

Create `.env` in project root:

```env
# JWT Secret (generate with: openssl rand -hex 32)
JWT_SECRET_KEY=your-super-secret-key-here-use-openssl-rand-hex-32

# Database paths (auto-created)
USERS_DB_PATH=data/users.db
LOGGING_DB_PATH=data/logging.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend URLs (for CORS)
USER_PORTAL_URL=http://localhost:3000
ADMIN_DASHBOARD_URL=http://localhost:3001
```

---

## 🔒 Security Notes

1. **Change default passwords** in production
2. **Use strong JWT_SECRET_KEY** (32+ random bytes)
3. **Enable HTTPS** in production
4. **Rate limit** authentication endpoints
5. **Rotate tokens** every 8 hours (configurable in `auth.py`)
6. **Hash passwords** with bcrypt (already implemented)
7. **Validate all inputs** (Pydantic schemas already in place)

---

## 📚 API Documentation

Once backend is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

All endpoints are documented with request/response schemas.

---

## 🐛 Troubleshooting

**Problem**: "Could not validate credentials"  
**Solution**: Token expired or invalid. Log in again.

**Problem**: "Admin privileges required"  
**Solution**: Use admin account (`admin` / `admin123`)

**Problem**: "Risk model not loaded"  
**Solution**: Train model first: `python scripts\train_simple_model.py --data your_data.csv --target target`

**Problem**: CORS errors in browser  
**Solution**: Check backend CORS middleware includes your frontend URL

---

## 📞 Support

For issues or questions, check:
- API docs: http://localhost:8000/docs
- Backend logs: Check terminal running uvicorn
- Database: Use SQLite browser to inspect `data/*.db` files

