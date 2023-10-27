#!/bin/bash
set -e
CONTAINER_ENGINE="docker"

if [ "$DEQLB_HOME" = "" ];
then
    # Error if path to dolfinx_eqlb i not set
    echo "Patch to source folder not set! Use "export DEQLB_HOME=/home/.../dolfinx_eqlb""
    exit 1
else
    # Build docker image
    echo "DEQLB_HOME is set to '$DEQLB_HOME'"
    ${CONTAINER_ENGINE} pull dolfinx/dolfinx:v0.6.0-r1
    ${CONTAINER_ENGINE} build --no-cache -f "${DEQLB_HOME}/docker/Dockerfile" -t brodbeck-m/dolfinx_eqlb:release .
fi
