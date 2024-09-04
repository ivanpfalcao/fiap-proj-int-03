#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)
NAMESPACE="trino"


kubectl -n "${NAMESPACE}" delete svc hive-metastore-service
kubectl -n "${NAMESPACE}" delete statefulset hive-metastore
kubectl -n "${NAMESPACE}" apply -f "${BASEDIR}/hive-metastore.yaml"