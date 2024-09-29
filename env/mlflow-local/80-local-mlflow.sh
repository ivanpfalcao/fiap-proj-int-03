BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

mkdir -p "${BASEDIR}/mlruns"

echo "sqlite://${BASEDIR}/mlflow.db"

mlflow server \
    --backend-store-uri "sqlite:///${BASEDIR}/mlflow.db" \
    --default-artifact-root "${BASEDIR}/mlruns" \
    --host 0.0.0.0 \
    --port 5000