#!/bin/bash

mlflow_track_server="http://user:pswd@10.96.132.192"

# Set the tracking URI
export MLFLOW_TRACKING_URI="$mlflow_track_server"

# Build the Docker image using mlflow models build-docker
mlflow models build-docker \
    --name "test_mlflow:1.0.0" \
    --model-uri "models:/teste_model/1" \
    --enable-mlserver