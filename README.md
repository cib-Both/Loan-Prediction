# 🏦 Loan Prediction System - Complete Setup Guide

A full-stack loan approval prediction system using Machine Learning, Flask backend, and modern HTML/CSS frontend.

## 📁 Project Structure

```
loan_prediction/
├── loan_approval_dataset.csv          # Training data
├── loan_prediction_model.ipynb        # ML model training notebook
├── loan_prediction_model.pkl          # Saved model (generated)
├── scaler.pkl                         # Saved scaler (generated)
├── feature_names.pkl                  # Saved features (generated)
├── README.md                          # This file
│
├── back-end/                          # Flask API
│   ├── app.py                        # Main Flask application
│   ├── requirements.txt              # Python dependencies
│   ├── test_api.py                   # API testing script
│   └── README.md                     # Backend documentation
│
└── front-end/                         # Web interface
    ├── index.html                    # Loan application form
    └── styles.css                    # Styling
```

## 🚀 Quick Start

### Step 1: Train the Model (if not done yet)

1. Open `loan_prediction_model.ipynb` in Jupyter/VS Code
2. Run all cells to train the model
3. This will generate three files:
   - `loan_prediction_model.pkl`
   - `scaler.pkl`
   - `feature_names.pkl`

### Step 2: Install Backend Dependencies

```bash
cd back-end
pip install -r requirements.txt
```

**Required packages:**
- Flask 3.0.0
- flask-cors 4.0.0
- numpy 1.26.0
- pandas 2.1.0
- scikit-learn 1.3.0
- joblib 1.3.2
- xgboost 2.0.0

### Step 3: Start the Backend Server

```bash
cd back-end
python app.py
```

The API will start on `http://localhost:5000`

You should see:
```
✅ Model and preprocessing objects loaded successfully!
✅ All systems ready!
📡 API running on http://localhost:5000
```

### Step 4: Open the Frontend

1. Navigate to the `front-end` folder
2. Open `index.html` in a web browser
3. Fill out the loan application form
4. Click "Predict Loan Approval"

## 🧪 Testing

### Test the API Endpoints

```bash
cd back-end
python test_api.py
```

This will test:
- ✅ Home endpoint
- ✅ Health check
- ✅ Model info
- ✅ Successful prediction (approved case)
- ✅ Successful prediction (rejected case)
- ❌ Error handling (missing fields)
- ❌ Error handling (invalid data)

### Manual API Testing with cURL

```bash
# Test health endpoint
curl http://localhost:5000/health

# Test prediction
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

## 📊 API Endpoints

### GET `/`
Check API status

### GET `/health`
Health check - verify model is loaded

### POST `/predict`
Make loan prediction

**Request body:**
```json
{
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
      "cibil_score": 720,
      "annual_income": "₹50,00,000",
      "loan_amount": "₹1,00,00,000",
      "loan_term": "10 years",
      "total_assets": "₹75,00,000"
    }
  }
}
```

### GET `/model-info`
Get model information

## 🎨 Features

### Frontend
- ✨ Modern, responsive design
- 📱 Mobile-friendly interface
- ✅ Real-time form validation
- 🎭 Smooth animations
- 📊 Detailed prediction results
- ⚡ Async API communication

### Backend
- 🔒 Input validation
- 🧠 ML model integration
- 📈 Confidence scores
- ⚠️ Risk analysis
- 🔄 CORS enabled
- 📝 Comprehensive error handling

## 🛠️ Troubleshooting

### Error: "Unable to connect to server"
**Solution:** Make sure the Flask backend is running
```bash
cd back-end
python app.py
```

### Error: "Model not loaded"
**Solution:** Train the model first by running the Jupyter notebook
1. Open `loan_prediction_model.ipynb`
2. Run all cells
3. Verify that `.pkl` files are created in the root directory

### Error: ModuleNotFoundError
**Solution:** Install required packages
```bash
cd back-end
pip install -r requirements.txt
```

### CORS Error in Browser
**Solution:** The `flask-cors` package should handle this. If issues persist:
1. Make sure `flask-cors` is installed
2. Check that CORS is enabled in `app.py`
3. Try accessing from `http://localhost` instead of `file://`

## 📦 Dependencies

### Python Packages
- Flask - Web framework
- flask-cors - CORS support
- scikit-learn - ML framework
- pandas - Data manipulation
- numpy - Numerical operations
- joblib - Model serialization
- xgboost - Gradient boosting

### Browser Requirements
- Modern browser with JavaScript enabled
- Supports ES6+ (Chrome, Firefox, Edge, Safari)

## 🔐 Security Notes

**For production deployment:**
1. Add authentication/authorization
2. Implement rate limiting
3. Add HTTPS support
4. Validate and sanitize all inputs
5. Use environment variables for configuration
6. Add logging and monitoring
7. Implement proper error handling
8. Add database for storing predictions

## 📈 Model Performance

The system uses the best performing model from:
- Logistic Regression
- Random Forest
- XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Check the notebook for detailed performance metrics.

## 🎯 Usage Example

1. **Start Backend:**
   ```bash
   cd back-end
   python app.py
   ```

2. **Open Frontend:**
   - Open `front-end/index.html` in browser

3. **Fill Form:**
   - Personal info (dependents, education, employment)
   - Financial info (income, loan amount, CIBIL score)
   - Assets info (residential, commercial, luxury, bank)

4. **Get Prediction:**
   - Click "Predict Loan Approval"
   - View result with confidence score
   - Check risk analysis and detailed summary

## 📝 License

This project is for educational purposes.

## 👥 Contributing

Feel free to submit issues and enhancement requests!

## 🙏 Acknowledgments

- Dataset: Loan approval prediction dataset
- ML Framework: scikit-learn, XGBoost
- Web Framework: Flask
- Frontend: HTML5, CSS3, JavaScript ES6+

---

**Happy Predicting! 🎉**
