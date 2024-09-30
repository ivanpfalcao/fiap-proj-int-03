#!/bin/bash

set -e
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

#export MLFLOW_TRACKING_URI="http://user:pswd@localhost:5000"
##export MLFLOW_MODEL_URI="models:/movie_rating_predictor/latest"
#export MLFLOW_MODEL_URI="mlflow-artifacts:/0/719b3d97d64e470ca28eda86a0308e52/artifacts/model_mlp_2024-09-30_03-34-02"




echo "Current date: $(date)"

streamlit run "${BASEDIR}/streamlit.py"

