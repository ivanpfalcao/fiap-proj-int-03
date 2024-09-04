#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)
NAMESPACE="datalake-ns"

kubectl -n "${NAMESPACE}" apply -f ${BASEDIR}/doriscluster.yaml