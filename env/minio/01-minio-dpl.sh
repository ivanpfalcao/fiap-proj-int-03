#!/bin/bash
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"

echo ${1}
NAMESPACE="datalake-ns"
if [ "$1" != "" ]
then
    NAMESPACE="${1}"
fi

kubectl -n "${NAMESPACE}" apply -k "${BASEDIR}/yamls"
