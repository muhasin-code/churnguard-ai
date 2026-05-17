#!/bin/bash
# View ChurnGuard AI logs

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the root of the project
cd "$(dirname "$0")/.."

# Verify Docker command availability
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed."
    exit 1
fi

# Detect if standard docker command works or if we require sudo
DOCKER_CMD="docker"
if ! docker info > /dev/null 2>&1; then
    if sudo docker info > /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
    else
        echo "Error: Cannot connect to the Docker daemon."
        exit 1
    fi
fi

# Route logs based on arguments
if [ "$1" == "mlflow" ]; then
    echo "Tailing MLflow logs..."
    $DOCKER_CMD compose logs -f mlflow
elif [ "$1" == "api" ]; then
    echo "Tailing API logs..."
    $DOCKER_CMD compose logs -f api
else
    echo "Tailing all service logs..."
    $DOCKER_CMD compose logs -f
fi