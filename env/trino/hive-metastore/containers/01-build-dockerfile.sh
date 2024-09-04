#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)

docker build \
    -f "${BASEDIR}/trino-hive-metastore.dockerfile" \
    -t "ivanpfalcao/hive:3.1.3" \
    --progress=plain \
    "${BASEDIR}/.."