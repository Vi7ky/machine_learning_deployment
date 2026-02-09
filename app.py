import numpy as np
from flask import request, jsonify
from flask import Flask
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.joblib')

# Load the StandardScaler
scaler = joblib.load('scaler.joblib')

print("Flask app initialized, model and scaler loaded.")
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)

    # Convert data to DataFrame, scale, and predict
    # Assuming the incoming JSON data matches the training features order
    # For simplicity, we are expecting a list of values for a single prediction
    # In a real-world scenario, robust validation and data structuring would be needed.
    features = pd.DataFrame([data])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': int(prediction[0])})

if __name__ == "__main__":
  print("Starting Prediction API with preprocessing and model inference...")
  app.run(debug=True)
