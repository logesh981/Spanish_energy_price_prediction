import os
import pandas as pd
import boto3
import mlflow
from dotenv import load_dotenv


load_dotenv()
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("us-east-1"),
)


def get_model_location(run_id):
    model_location = os.getenv('MODEL_LOCATION')

    if model_location is not None:
        return model_location

    model_bucket = os.getenv('MODEL_BUCKET', 'mlflow-artifacts-remote23')
    experiment_id = os.getenv('MLFLOW_EXPERIMENT_ID', '839689426251288043')

    model_location = (
        f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models_mlflow'
    )

    return model_location


def load_model(run_id):
    model_path = get_model_location(run_id)
    model = mlflow.xgboost.load_model(model_path)
    return model


class ModelService:
    def __init__(self, model, model_version=None):

        self.model = model
        self.model_version = model_version

    def prepare_features(self, features):

        features['total_fossil_generation'] = (
            features['generation_fossil_hard_coal']
            + features['generation_fossil_gas']
            + features['generation_fossil_oil']
            + features['generation_fossil_brown_coal/lignite']
        )

        return features

    def predict(self, features):
        # convert the json received in the request to a list of dict for the model input
        if isinstance(features, dict):
            features = [features]
        pd_features = pd.DataFrame(features)
        pred = self.model.predict(pd_features)
        return pred
