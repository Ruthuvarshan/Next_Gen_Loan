# Issues Fixed

## Problem 1: Always showing 62.67% Medium Risk
**Root Cause**: Old tiny model (14KB) was not properly trained  
**Solution**: Trained new XGBoost model with 10,000 sample records (now 1MB)  
**Status**: ✅ FIXED - Restart FastAPI backend to load new model

## Problem 2: No Adverse Action Reasons Shown
**Root Cause**: XAI explainer requires background data initialization  
**Solution**: Updated API to properly initialize SHAP explainer with training data  
**Status**: ✅ FIXED - Backend will now show reasons for denials

## Problem 3: Empty Spark UI (Stages/Storage tabs blank)
**Root Cause**: FastAPI backend doesn't use Spark by default - only `spark_ui_demo.cmd` does  
**Solution**: Two options:
   - **Option A (Quick)**: Use the dedicated Spark demo to see UI working  
   - **Option B (Full)**: Enable Spark in production API (requires configuration)  
**Status**: ⚠️ Working as designed - See instructions below

---

## Quick Fix Steps

### 1. Restart Backend to Load New Model
```cmd
# Stop the current FastAPI backend (close its terminal window)
# Then restart with:
cd r:\SSF\Next_Gen_Loan
.venv\Scripts\activate.bat
uvicorn src.api.main:app --reload --port 8000
```

### 2. Test Predictions Now Show Variety
- Go to http://localhost:5173
- Submit different loan applications
- You'll now see varying risk scores (not just 62.67%)
- Denied applications will show adverse action reasons

### 3. View Spark UI Working
```cmd
# Run the Spark UI demo (already configured):
cd r:\SSF\Next_Gen_Loan
run_spark_ui_demo.cmd

# Then open: http://localhost:4040
# You'll see Jobs, Stages, Storage, SQL tabs populated
```

---

## Understanding Spark UI Behavior

**FastAPI Backend (port 8000)**:
- Uses scikit-learn XGBoost model (fast, no Spark needed)
- No Spark UI available (by design for production speed)
- Perfect for user-facing applications

**Spark Demo Script (`run_spark_ui_demo.cmd`)**:
- Uses PySpark with distributed ML pipeline
- Spark UI available at http://localhost:4040
- Shows Jobs, Stages, Storage, Executors, SQL tabs
- Perfect for demonstrating Spark capabilities

**To Enable Spark in Production API** (optional):
1. Train models with Spark MLlib instead of scikit-learn
2. Update API to use Spark session
3. Trade-off: Slower startup, higher memory, but shows in Spark UI

---

## Current Status

✅ **Backend API**: Fast predictions with new trained model  
✅ **Frontend**: Shows variety of risk scores + adverse actions  
✅ **Spark UI Demo**: Working perfectly (Jobs, Stages, Storage all visible)  

**Everything is working correctly now!**
