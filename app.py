from flask import Flask, request, jsonify
from joblib import load
import numpy as np

model_dict = load("diabetes_pipeline.joblib")
model = model_dict["pipeline"]
FEATURES = model_dict["features"]


app = Flask(__name__)

@app.route("/")
def home():
    return "Diabetes Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        features = [data[feat] for feat in FEATURES]
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400

    features = np.array([features])

    prediction = model.predict(features)[0]

    result = {
        "prediction": int(prediction),   # 0 = No Diabetes, 1 = Diabetes
        "message": "Diabetes Detected" if prediction == 1 else "No Diabetes"
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
