#!/bin/bash

set -e
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

export TMDB_TOKEN="eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NWFlZjk4ZWVkNTM1NWRmYmUyNGVjNmZiOWU2ZGZjOCIsIm5iZiI6MTcyNjk0NTM5Mi4zMzgyOCwic3ViIjoiNjZlZjE3NzI2YzNiN2E4ZDY0OGQzYWM1Iiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.Y45bt_CPC9FnKCAPe4T2-nYnA3u48ZE6Hmm912zxzLs"
export OUTPUT_MOVIES_FOLDER="${BASEDIR}/output_folder"
export TMDB_NUMBER_OF_PAGES="100"

export LOCAL_FOLDER="${OUTPUT_MOVIES_FOLDER}"
export MINIO_BUCKET="datalakebucket"
export MINIO_BUCKET_FOLDER="raw/movies"
export MINIO_ENDPOINT="http://localhost:9000"
export MINIO_ACCESS_KEY="minio"
export MINIO_SECRET_KEY="minio123"
export MINIO_USE_SSL="false"


echo "Current date: $(date)"

mkdir -p "${LOCAL_FOLDER}"

#python "${BASEDIR}/get_movies.py"

python "${BASEDIR}/transfer_data.py"

