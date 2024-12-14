import os
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import model


def read_json(filepath):
    test_directory = Path(__file__).parent
    with open(test_directory / filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def test_prepare_features():
    model_service = model.ModelService(None)
    # features =pd.read_json("test.json")
    json_data = read_json("test.json")

    modified_features = model_service.prepare_features(json_data)

    expected_fetures = {
        "generation_biomass": 447.0,
        "generation_fossil_brown_coal/lignite": 329.0,
        "generation_fossil_gas": 4844.0,
        "generation_fossil_hard_coal": 4821.0,
        "generation_fossil_oil": 162.0,
        "generation_hydro_pumped_storage_consumption": 863.0,
        "generation_hydro_run-of-river_and_poundage": 1051.0,
        "generation_hydro_water_reservoir": 1899.0,
        "generation_nuclear": 7096.0,
        "generation_other": 43.0,
        "generation_other_renewable": 73.0,
        "generation_solar": 49.0,
        "generation_waste": 196.0,
        "generation_wind_onshore": 6378.0,
        "forecast_solar_day_ahead": 17.0,
        "forecast_wind_onshore_day_ahead": 6436.0,
        "total_load_forecast": 26118.0,
        "total_load_actual": 25385.0,
        "price_day_ahead": 50.1,
        "total_fossil_generation": 10156,
    }

    assert modified_features == expected_fetures


def test_predict():
    load_dotenv()
    run_id = os.getenv("MLFLOW_RUN_ID")
    test_model = model.load_model(run_id)
    model_service = model.ModelService(test_model)
    data = read_json("test.json")
    modified_features = model_service.prepare_features(data)
    prediction = model_service.predict(modified_features)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1,)
    assert prediction[0] > 1
