# Loan Prediction API - Backend

Flask-based REST API for loan approval prediction using machine learning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Server

```bash
python app.py
```

The API will start on `http://localhost:5000`

## 📡 API Endpoints

### GET `/`
Check API status and available endpoints.

**Response:**
```json
{
  "status": "success",
  "message": "🏦 Loan Prediction API is running!",
  "endpoints": {
    "/": "API status",
    "/predict": "POST - Make loan prediction",
    "/health": "Health check"
  }
}
```

### GET `/health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### POST `/predict`
Make loan approval prediction.

**Request Body:**
```json
{
  "no_of_dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "income_annum": 9600000,
  "loan_amount": 29900000,
  "loan_term": 12,
  "cibil_score": 778,
  "residential_assets_value": 2400000,
  "commercial_assets_value": 17600000,
  "luxury_assets_value": 22700000,
  "bank_asset_value": 8000000
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Approved",
  "confidence": 95.67,
  "details": {
    "loan_status": "Approved",
    "approval_probability": 95.67,
    "rejection_probability": 4.33,
    "risk_factors": ["No major risk factors identified"],
    "applicant_summary": {
      "cibil_score": 778,
      "annual_income": "₹96,00,000",
      "loan_amount": "₹2,99,00,000",
      "loan_term": "12 years",
      "total_assets": "₹5,07,00,000"
    }
  }
}
```

### GET `/model-info`
Get information about the loaded model.

**Response:**
```json
{
  "status": "success",
  "model_type": "RandomForestClassifier",
  "features": ["no_of_dependents", "education", "self_employed", ...],
  "num_features": 11
}
```

## 🔧 Configuration

The API expects the following files in the parent directory:
- `loan_prediction_model.pkl` - Trained ML model
- `scaler.pkl` - Feature scaler
- `feature_names.pkl` - List of feature names

## 📝 Field Validation

**Required Fields:**
- `no_of_dependents`: Number (0-10)
- `education`: String ("Graduate" or "Not Graduate")
- `self_employed`: String ("Yes" or "No")
- `income_annum`: Number (annual income in rupees)
- `loan_amount`: Number (requested loan amount)
- `loan_term`: Number (loan duration in years)
- `cibil_score`: Number (300-900)
- `residential_assets_value`: Number
- `commercial_assets_value`: Number
- `luxury_assets_value`: Number
- `bank_asset_value`: Number

## 🧪 Testing the API

### Using cURL:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "no_of_dependents": 1,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5000000,
    "loan_amount": 10000000,
    "loan_term": 10,
    "cibil_score": 720,
    "residential_assets_value": 3000000,
    "commercial_assets_value": 1000000,
    "luxury_assets_value": 1500000,
    "bank_asset_value": 2000000
  }'
```

### Using Python:

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "no_of_dependents": 1,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 5000000,
    "loan_amount": 10000000,
    "loan_term": 10,
    "cibil_score": 720,
    "residential_assets_value": 3000000,
    "commercial_assets_value": 1000000,
    "luxury_assets_value": 1500000,
    "bank_asset_value": 2000000
}

response = requests.post(url, json=data)
print(response.json())
```

## ⚠️ Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (missing/invalid fields)
- `404`: Endpoint not found
- `500`: Internal server error

## 🔒 CORS

CORS is enabled to allow frontend applications to communicate with the API.

## 📦 Dependencies

- Flask: Web framework
- flask-cors: CORS support
- scikit-learn: ML model support
- pandas: Data manipulation
- numpy: Numerical operations
- joblib: Model serialization
- xgboost: XGBoost model support
