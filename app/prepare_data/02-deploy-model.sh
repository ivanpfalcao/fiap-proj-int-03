#!/bin/bash
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"


NAMESPACE="mlflow-ns"
if [ "$1" != "" ]
then
    NAMESPACE="${1}"
fi
echo ${NAMESPACE}

kubectl -n "${NAMESPACE}" apply -f "${BASEDIR}/deploy_model.yaml"