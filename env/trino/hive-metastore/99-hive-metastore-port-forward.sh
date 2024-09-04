#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)
NAMESPACE="trino"

kubectl -n "${NAMESPACE}" port-forward svc/hive-metastore-service 9083:9083 10000:10000
