#!/bin/bash
BASEDIR="$( cd "$( dirname "${0}" )" && pwd )"


NAMESPACE="datalake-ns"
if [ "$1" != "" ]
then
    NAMESPACE="${1}"
fi
echo ${NAMESPACE}


kubectl -n ${NAMESPACE} create secret generic tmdb-token-secret --from-literal=TMDB_TOKEN="eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1NWFlZjk4ZWVkNTM1NWRmYmUyNGVjNmZiOWU2ZGZjOCIsIm5iZiI6MTcyNjk0NTM5Mi4zMzgyOCwic3ViIjoiNjZlZjE3NzI2YzNiN2E4ZDY0OGQzYWM1Iiwic2NvcGVzIjpbImFwaV9yZWFkIl0sInZlcnNpb24iOjF9.Y45bt_CPC9FnKCAPe4T2-nYnA3u48ZE6Hmm912zxzLs"