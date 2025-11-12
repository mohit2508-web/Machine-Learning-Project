from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODELS_DIR = "models"

try:
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    return jsonify({
        "message": "Telco Churn Prediction API",
        "status": "running",
        "model_loaded": model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.json
        
        input_data = pd.DataFrame([{
            'gender': data.get('gender', 'Male'),
            'SeniorCitizen': int(data.get('SeniorCitizen', 0)),
            'Partner': data.get('Partner', 'No'),
            'Dependents': data.get('Dependents', 'No'),
            'tenure': float(data.get('tenure', 12)),
            'PhoneService': data.get('PhoneService', 'Yes'),
            'MultipleLines': data.get('MultipleLines', 'No'),
            'InternetService': data.get('InternetService', 'Fiber optic'),
            'OnlineSecurity': data.get('OnlineSecurity', 'No'),
            'OnlineBackup': data.get('OnlineBackup', 'No'),
            'DeviceProtection': data.get('DeviceProtection', 'No'),
            'TechSupport': data.get('TechSupport', 'No'),
            'StreamingTV': data.get('StreamingTV', 'No'),
            'StreamingMovies': data.get('StreamingMovies', 'No'),
            'Contract': data.get('Contract', 'Month-to-month'),
            'PaperlessBilling': data.get('PaperlessBilling', 'Yes'),
            'PaymentMethod': data.get('PaymentMethod', 'Electronic check'),
            'MonthlyCharges': float(data.get('MonthlyCharges', 70)),
            'TotalCharges': float(data.get('TotalCharges', 840))
        }])
        
        from sklearn.preprocessing import LabelEncoder
        
        for col in input_data.select_dtypes(include=['object']).columns:
            if input_data[col].nunique() == 2:
                le = LabelEncoder()
                le.fit(['Yes', 'No'] if col != 'gender' else ['Male', 'Female'])
                input_data[col] = le.transform(input_data[col])
            else:
                input_data = pd.get_dummies(input_data, columns=[col], drop_first=True)
        
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0
        
        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        churn_prob = probability[1]
        if churn_prob > 0.7:
            risk_level = "High"
        elif churn_prob > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return jsonify({
            "willChurn": bool(prediction),
            "probability": float(churn_prob),
            "confidence": float(max(probability)),
            "riskLevel": risk_level,
            "probabilities": {
                "stay": float(probability[0]),
                "churn": float(probability[1])
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/model-info', methods=['GET'])
def model_info():
    return jsonify({
        "model_type": type(model).__name__ if model else None,
        "features_count": len(feature_names) if feature_names else 0,
        "features": feature_names if feature_names else []
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)