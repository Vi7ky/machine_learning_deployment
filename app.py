import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

print("Flask app initialized, model and scaler loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Convert input JSON to DataFrame
    features = pd.DataFrame([data])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
    print("Starting Prediction API...")
    app.run(debug=True)
