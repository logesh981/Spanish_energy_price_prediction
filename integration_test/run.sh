#!/usr/bin/env bash

cd "$(dirname "$0")"
LOCAL_TAG=`date +"%Y-%m-%d-%H-%M"`
export LOCAL_IMAGE_NAME="lokidocker98/energy-price-prediction:${LOCAL_TAG}"

docker build -t ${LOCAL_IMAGE_NAME} ..

docker-compose up -d

pipenv run python test_docker.py

ERROR_CODE=$?

if [ ${ERROR_CODE} != 0 ]; then
    docker-compose logs
    docker-compose down
    exit ${ERROR_CODE}
fi

docker-compose down
