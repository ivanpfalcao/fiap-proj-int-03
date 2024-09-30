#!/bin/bash
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"


NAMESPACE="datalake-ns"
if [ "$1" != "" ]
then
    NAMESPACE="${1}"
fi
echo ${NAMESPACE}

kubectl -n "${NAMESPACE}" apply -f "${BASEDIR}/streamlit.yaml"