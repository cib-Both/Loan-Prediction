from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the saved model and preprocessing objects
MODEL_PATH = '../loan_prediction_model.pkl'
SCALER_PATH = '../scaler.pkl'
FEATURES_PATH = '../feature_names.pkl'

# Global variables to store loaded objects
model = None
scaler = None
feature_names = None

def load_model():
    """Load the saved model and preprocessing objects"""
    global model, scaler, feature_names
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        print("✅ Model and preprocessing objects loaded successfully!")
        print(f"📋 Expected features: {feature_names}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

def preprocess_input(data):
    """
    Preprocess input data to match training format with feature engineering
    """
    # Create a copy to avoid modifying original
    input_data = data.copy()
    
    # Encoding categorical variables (same as training)
    # Education: Graduate = 1, Not Graduate = 0
    if isinstance(input_data.get('education'), str):
        input_data['education'] = 1 if input_data['education'] == 'Graduate' else 0
    
    # Self Employed: Yes = 1, No = 0
    if isinstance(input_data.get('self_employed'), str):
        input_data['self_employed'] = 1 if input_data['self_employed'] == 'Yes' else 0
    
    # Calculate engineered features (CRITICAL - these were in training!)
    # 1. Total Assets
    input_data['total_assets'] = (
        input_data['residential_assets_value'] + 
        input_data['commercial_assets_value'] + 
        input_data['luxury_assets_value'] + 
        input_data['bank_asset_value']
    )
    
    # 2. Loan to Income Ratio
    input_data['loan_to_income_ratio'] = input_data['loan_amount'] / input_data['income_annum']
    
    # 3. Loan to Asset Ratio
    input_data['loan_to_asset_ratio'] = input_data['loan_amount'] / (input_data['total_assets'] + 1)
    
    # 4. Monthly EMI Estimate
    input_data['monthly_emi_estimate'] = input_data['loan_amount'] / (input_data['loan_term'] * 12)
    
    # 5. EMI to Monthly Income Ratio
    input_data['emi_to_monthly_income'] = input_data['monthly_emi_estimate'] / (input_data['income_annum'] / 12)
    
    # 6. Income per Dependent
    input_data['income_per_dependent'] = input_data['income_annum'] / (input_data['no_of_dependents'] + 1)
    
    # 7. CIBIL Score Category
    cibil = input_data['cibil_score']
    if cibil <= 500:
        input_data['cibil_category'] = 0
    elif cibil <= 650:
        input_data['cibil_category'] = 1
    elif cibil <= 750:
        input_data['cibil_category'] = 2
    else:
        input_data['cibil_category'] = 3
    
    # Create dataframe with all features
    df = pd.DataFrame([input_data])
    
    # Ensure all columns are present and in correct order
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Select only the features used in training, in correct order
    df = df[feature_names]
    
    # NO SCALING - Random Forest doesn't need it!
    # The model was trained on unscaled data
    return df.values

@app.route('/')
def home():
    """Home endpoint to check if API is running"""
    return jsonify({
        'status': 'success',
        'message': '🏦 Loan Prediction API is running!',
        'endpoints': {
            '/': 'API status',
            '/predict': 'POST - Make loan prediction',
            '/health': 'Health check'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict loan approval based on input features
    
    Expected JSON format:
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
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No data provided'
            }), 400
        
        # Validate required fields
        required_fields = [
            'no_of_dependents', 'education', 'self_employed', 
            'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
            'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Convert numeric fields to appropriate types
        numeric_fields = [
            'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value'
        ]
        
        for field in numeric_fields:
            try:
                data[field] = float(data[field])
            except (ValueError, TypeError):
                return jsonify({
                    'status': 'error',
                    'message': f'Invalid value for {field}. Must be a number.'
                }), 400
        
        # Preprocess the input
        processed_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        prediction_proba = model.predict_proba(processed_data)[0]
        
        # Convert prediction to human-readable format
        result = 'Approved' if prediction == 1 else 'Rejected'
        confidence = float(prediction_proba[1] if prediction == 1 else prediction_proba[0])
        
        # Calculate comprehensive risk factors and positive factors
        risk_factors = []
        positive_factors = []
        
        # Calculate derived metrics
        total_assets = (data['residential_assets_value'] + data['commercial_assets_value'] + 
                       data['luxury_assets_value'] + data['bank_asset_value'])
        loan_to_income_ratio = data['loan_amount'] / data['income_annum'] if data['income_annum'] > 0 else 0
        
        # Credit Score Analysis
        if data['cibil_score'] >= 750:
            positive_factors.append('Excellent CIBIL score (750+)')
        elif data['cibil_score'] >= 700:
            positive_factors.append('Good CIBIL score (700-749)')
        elif data['cibil_score'] >= 650:
            positive_factors.append('Fair CIBIL score (650-699)')
        else:
            risk_factors.append(f'Low CIBIL score ({int(data["cibil_score"])})')
        
        # Income Analysis (in dollars)
        if data['income_annum'] >= 100000:
            positive_factors.append(f'Strong annual income (${int(data["income_annum"]):,})')
        elif data['income_annum'] >= 50000:
            positive_factors.append(f'Moderate annual income (${int(data["income_annum"]):,})')
        else:
            risk_factors.append(f'Low annual income (${int(data["income_annum"]):,})')
        
        # Loan to Income Ratio
        if loan_to_income_ratio > 5:
            risk_factors.append(f'Very high loan-to-income ratio ({loan_to_income_ratio:.1f}x)')
        elif loan_to_income_ratio > 3:
            risk_factors.append(f'High loan-to-income ratio ({loan_to_income_ratio:.1f}x)')
        elif loan_to_income_ratio <= 2:
            positive_factors.append(f'Healthy loan-to-income ratio ({loan_to_income_ratio:.1f}x)')
        
        # Asset Coverage
        if total_assets > data['loan_amount'] * 1.5:
            positive_factors.append(f'Strong asset coverage (${int(total_assets):,})')
        elif total_assets < data['loan_amount'] * 0.5:
            risk_factors.append(f'Insufficient asset coverage (${int(total_assets):,})')
        
        # Loan Term Analysis
        if data['loan_term'] > 20:
            risk_factors.append(f'Long loan term ({int(data["loan_term"])} years)')
        elif data['loan_term'] <= 15:
            positive_factors.append(f'Manageable loan term ({int(data["loan_term"])} years)')
        
        # Dependents
        if data['no_of_dependents'] > 4:
            risk_factors.append(f'High number of dependents ({int(data["no_of_dependents"])})')
        elif data['no_of_dependents'] <= 2:
            positive_factors.append(f'Low dependency ratio ({int(data["no_of_dependents"])})')
        
        # Employment Status
        if data['self_employed'] == 1 or str(data.get('self_employed', '')).lower() == 'yes':
            risk_factors.append('Self-employed (variable income)')
        else:
            positive_factors.append('Salaried employment (stable income)')
        
        # Education
        if data['education'] == 1 or str(data.get('education', '')).lower() == 'graduate':
            positive_factors.append('Graduate education level')
        
        # Final risk assessment
        if not risk_factors:
            risk_factors = ['No significant risk factors - Strong application']
        
        if not positive_factors:
            positive_factors = ['Application meets minimum requirements']
        
        # Prepare response
        response = {
            'status': 'success',
            'prediction': result,
            'confidence': round(confidence * 100, 2),
            'details': {
                'loan_status': result,
                'approval_probability': round(prediction_proba[1] * 100, 2),
                'rejection_probability': round(prediction_proba[0] * 100, 2),
                'risk_factors': risk_factors,
                'positive_factors': positive_factors,
                'applicant_summary': {
                    'cibil_score': data['cibil_score'],
                    'annual_income': f"${int(data['income_annum']):,}",
                    'loan_amount': f"${int(data['loan_amount']):,}",
                    'loan_term': f"{int(data['loan_term'])} years",
                    'total_assets': f"${int(total_assets):,}",
                    'loan_to_income_ratio': f"{loan_to_income_ratio:.2f}x"
                }
            }
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/model-info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 500
    
    return jsonify({
        'status': 'success',
        'model_type': type(model).__name__,
        'features': feature_names,
        'num_features': len(feature_names)
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting Loan Prediction API...")
    
    # Load the model
    if load_model():
        print("✅ All systems ready!")
        print("📡 API running on http://localhost:5000")
        print("\nAvailable endpoints:")
        print("   GET  /           - API status")
        print("   GET  /health     - Health check")
        print("   POST /predict    - Make prediction")
        print("   GET  /model-info - Model information")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model. Please ensure model files exist:")
        print(f"   - {MODEL_PATH}")
        print(f"   - {SCALER_PATH}")
        print(f"   - {FEATURES_PATH}")
