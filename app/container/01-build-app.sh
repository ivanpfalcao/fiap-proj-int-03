#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)

docker build \
    -f "${BASEDIR}/app.dockerfile" \
    -t "ivanpfalcao/fiap-dataprep-03:1.0.0" \
    --progress=plain \
    "${BASEDIR}/.."