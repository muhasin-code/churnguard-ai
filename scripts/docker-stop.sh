#!/bin/bash
# Stop ChurnGuard AI Docker containers

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the root of the project
cd "$(dirname "$0")/.."

echo "======================================================================="
echo "ChurnGuard AI - Stopping Docker Containers"
echo "======================================================================="

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

# Stop containers
$DOCKER_CMD compose down

echo ""
echo "======================================================================="
echo "Containers Stopped Successfully"
echo "======================================================================="
echo ""
echo "Additional Commands:"
echo "  Remove data volumes:   $DOCKER_CMD compose down -v"
echo "  Delete builder image:  $DOCKER_CMD image rm churnguard-ai-api"
echo ""