#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)
NAMESPACE="trino"

kubectl -n "${NAMESPACE}" delete job hive-metastore-job
kubectl -n "${NAMESPACE}" apply -f "${BASEDIR}/hive-init-postgres.yaml"