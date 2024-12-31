import json
from pathlib import Path
import requests


def read_json(filepath):
    test_directory = Path(__file__).parent
    with open(test_directory / filepath, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


json_data = read_json("test.json")

url = 'http://127.0.0.1:9696/predict'
response = requests.post(url, json=json_data, verify=False, timeout=10)

assert 1 == 1
