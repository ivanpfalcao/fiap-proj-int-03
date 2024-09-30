#!/bin/bash
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"


NAMESPACE="datalake-ns"
if [ "$1" != "" ]
then
    NAMESPACE="${1}"
fi
echo ${NAMESPACE}

kubectl -n "${NAMESPACE}" apply -f "${BASEDIR}/train-model.yaml"

kubectl -n "${NAMESPACE}" create job --from=cronjob/model-training-cronjob model-training-job-manual