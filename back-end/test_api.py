"""
Test script for Loan Prediction API
Run this after starting the Flask server to test the endpoints
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_home():
    """Test the home endpoint"""
    print("\n" + "="*60)
    print("Testing GET / endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_health():
    """Test the health endpoint"""
    print("\n" + "="*60)
    print("Testing GET /health endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Test the model info endpoint"""
    print("\n" + "="*60)
    print("Testing GET /model-info endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_approved():
    """Test prediction with likely approved case"""
    print("\n" + "="*60)
    print("Testing POST /predict - Likely APPROVED case")
    print("="*60)
    
    data = {
        "no_of_dependents": 1,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 9600000,
        "loan_amount": 15000000,
        "loan_term": 12,
        "cibil_score": 780,
        "residential_assets_value": 5000000,
        "commercial_assets_value": 10000000,
        "luxury_assets_value": 8000000,
        "bank_asset_value": 6000000
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_rejected():
    """Test prediction with likely rejected case"""
    print("\n" + "="*60)
    print("Testing POST /predict - Likely REJECTED case")
    print("="*60)
    
    data = {
        "no_of_dependents": 5,
        "education": "Not Graduate",
        "self_employed": "Yes",
        "income_annum": 2000000,
        "loan_amount": 15000000,
        "loan_term": 5,
        "cibil_score": 450,
        "residential_assets_value": 500000,
        "commercial_assets_value": 0,
        "luxury_assets_value": 200000,
        "bank_asset_value": 300000
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_missing_fields():
    """Test prediction with missing fields"""
    print("\n" + "="*60)
    print("Testing POST /predict - Missing fields (should fail)")
    print("="*60)
    
    data = {
        "no_of_dependents": 1,
        "education": "Graduate",
        "income_annum": 5000000
        # Missing other required fields
    }
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_invalid_data():
    """Test prediction with invalid data types"""
    print("\n" + "="*60)
    print("Testing POST /predict - Invalid data types (should fail)")
    print("="*60)
    
    data = {
        "no_of_dependents": "invalid",  # Should be number
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
    
    print(f"Input Data: {json.dumps(data, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def run_all_tests():
    """Run all test cases"""
    print("\n" + "🧪 " + "="*58)
    print("🧪 STARTING API TESTS")
    print("🧪 " + "="*58)
    
    try:
        test_home()
        test_health()
        test_model_info()
        test_predict_approved()
        test_predict_rejected()
        test_predict_missing_fields()
        test_predict_invalid_data()
        
        print("\n" + "✅ " + "="*58)
        print("✅ ALL TESTS COMPLETED")
        print("✅ " + "="*58 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to the API.")
        print("Make sure the Flask server is running on http://localhost:5000")
        print("Start it with: python app.py\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")

if __name__ == "__main__":
    run_all_tests()
