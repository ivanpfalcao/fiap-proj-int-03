#!/bin/bash

set -e
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

export TMDB_TOKEN="eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NWFlZjk4ZWVkNTM1NWRmYmUyNGVjNmZiOWU2ZGZjOCIsIm5iZiI6MTcyNjk0NTM5Mi4zMzgyOCwic3ViIjoiNjZlZjE3NzI2YzNiN2E4ZDY0OGQzYWM1Iiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.Y45bt_CPC9FnKCAPe4T2-nYnA3u48ZE6Hmm912zxzLs"
export OUTPUT_MOVIES_FOLDER="${BASEDIR}/output_folder"
export TMDB_NUMBER_OF_PAGES="100"

export LOCAL_FOLDER="${OUTPUT_MOVIES_FOLDER}"
export S3_BUCKET="datalakebucket"
export S3_RAW_MOVIE_FOLDER="s3://datalakebucket/bronze/movies/*.json"
export S3_SILVER_MOVIE_FOLDER="s3://datalakebucket/silver/movies"
export S3_GOLD_MODEL_MOVIE_FOLDER="s3://datalakebucket/gold/model_movies"
export S3_ENDPOINT="localhost:9000"
export S3_ACCESS_KEY="minio"
export S3_SECRET_KEY="minio123"
export S3_USE_SSL="false"


echo "Current date: $(date)"

python "${BASEDIR}/data_prep.py"

