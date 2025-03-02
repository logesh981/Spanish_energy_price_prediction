import os
import logging

from flask import Flask, request, jsonify
import model

# with open('xgmodel.bin','rb') as f_in:
#     model = pickle.load(f_in)
# def predict(features):
#     preds = model.predict(features)
#     return float(preds[0])


app = Flask('energy-price-prediction')

RUN_ID = os.getenv("MLFLOW_RUN_ID")
if RUN_ID is None:
    raise ValueError("MLFLOW_RUN_ID is not set in the environment variables")

try:
    modelt = model.load_model(RUN_ID)
    model_service = model.ModelService(modelt)
except Exception as e:
    logging.error("Failed to load the model: %s", str(e))
    raise e


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    energy_val = request.get_json()

    # Validate input
    if not isinstance(energy_val, dict):
        return jsonify({"error": "Invalid input format. Expected JSON."}), 400

    modified_features = model_service.prepare_features(energy_val)
    prediction = model_service.predict(modified_features)

    result = {"pred": float(prediction[0])}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
