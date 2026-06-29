from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json

app = Flask(__name__)
CORS(app)

# load model
model = joblib.load("models/best_pipeline.pkl")

# load training columns
with open("models/features.json", "r") as f:
    FEATURES = json.load(f)


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'healthy (sklearn pipeline)'}


@app.route('/predict', methods=['POST'])
def predict():

    try:
        data = request.json

        # convert input
        input_df = pd.DataFrame([data])

        # 🔥 fill missing columns
        for col in FEATURES:
            if col not in input_df.columns:
                input_df[col] = 0  # default value

        # reorder columns
        input_df = input_df[FEATURES]

        # prediction
        prob = model.predict_proba(input_df)[0][1]
        pred = int(model.predict(input_df)[0])

        return jsonify({
            "prediction": pred,
            "probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)