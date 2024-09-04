#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)
NAMESPACE="datalake-ns"

helm repo add doris-repo https://charts.selectdb.com

kubectl create namespace ${NAMESPACE}
helm -n ${NAMESPACE} install operator doris-repo/doris-operator