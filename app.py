
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the best model selected in Experiment 2
model = joblib.load("model/heatwave_model.pkl")

FEATURES = [
    "max_temp",
    "min_temp",
    "humidity",
    "wind_speed",
    "previous_temp"
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heatwave Prediction API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "JSON object required"}), 400

    try:
        values = [float(data[key]) for key in FEATURES]
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide all five numeric features"
        }), 400

    input_data = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(input_data)[0]

    result = "Heatwave" if prediction == 1 else "No Heatwave"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
