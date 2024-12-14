import os


from flask import Flask, request, jsonify
import model


# with open('xgmodel.bin','rb') as f_in:
#     model = pickle.load(f_in)
# def predict(features):
#     preds = model.predict(features)
#     return float(preds[0])


app = Flask('energy-price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    energy_val = request.get_json()
    run_id = os.getenv("MLFLOW_RUN_ID")
    modelt = model.load_model(run_id)
    model_service = model.ModelService(modelt)
    modified_features = model_service.prepare_features(energy_val)
    prediction = model_service.predict(modified_features)
    # print("")

    # return jsonify(result)

    result = {"pred": float(prediction[0])}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
