#!/bin/bash

set -e
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

export MLFLOW_TRACKING_URI="http://user:pswd@localhost:5000"
export MLFLOW_MODEL_URI="models:/movie_rating_predictor/1"


echo "Current date: $(date)"

streamlit run "${BASEDIR}/streamlit.py"

