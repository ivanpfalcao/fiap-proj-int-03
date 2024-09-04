# Define the namespace for Kubernetes deployment (if not already set)
NAMESPACE="datalake-ns"

# Define the base directory relative to the script location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)


kubectl -n "${NAMESPACE}" apply -f "${BASEDIR}/postgres.yaml"