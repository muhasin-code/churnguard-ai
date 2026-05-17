#!/bin/bash
# Start ChurnGuard AI in Docker

# Exit immediately if a command exits with a non-zero status
set -e

# Change directory to the root of the project
cd "$(dirname "$0")/.."

echo "======================================================================="
echo "ChurnGuard AI - Starting Docker Containers"
echo "======================================================================="

# Verify Docker command availability
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed. Please install Docker first."
    exit 1
fi

# Detect if standard docker command works or if we require sudo
DOCKER_CMD="docker"
if ! docker info > /dev/null 2>&1; then
    echo "Access to Docker daemon requires elevated privileges."
    echo "Testing sudo access..."
    if sudo docker info > /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        echo "Using 'sudo docker' for container management."
    else
        echo "Error: Cannot connect to the Docker daemon."
        echo "Please verify that Docker is running and you have sufficient permissions."
        exit 1
    fi
fi

# Build images if they do not exist
if [[ "$($DOCKER_CMD images -q churnguard-ai-api 2> /dev/null)" == "" ]]; then
    echo "Building Docker images..."
    $DOCKER_CMD compose build
fi

# Start MLflow first to ensure it's fully ready before starting API
echo "Starting MLflow service..."
$DOCKER_CMD compose up -d mlflow

# Wait for services to be ready
echo ""
echo "Waiting for services to be ready..."
echo "Note: The first start may take longer as MLflow builds its environment."

# Dynamic wait for MLflow container to be healthy according to Docker daemon or responsive via HTTP
max_wait_mlflow=300
count_mlflow=0
echo -n "Waiting for MLflow tracked server to become healthy..."
while [ "$($DOCKER_CMD inspect --format='{{.State.Health.Status}}' churnguard-mlflow 2>/dev/null)" != "healthy" ] && ! curl -s http://localhost:5000/health > /dev/null 2>&1 && [ $count_mlflow -lt $max_wait_mlflow ]; do
    echo -n "."
    sleep 2
    count_mlflow=$((count_mlflow + 2))
done
echo ""

# Sleep briefly to ensure Docker DNS fully registers the container name
sleep 3

# Start the API service now that MLflow is healthy
echo "Starting ChurnGuard API service..."
$DOCKER_CMD compose up -d api

# Dynamic wait for ChurnGuard API (up to 60 seconds)
max_wait_api=60
count_api=0
echo -n "Waiting for ChurnGuard API (http://localhost:8000)..."
while ! curl -s http://localhost:8000/health/live > /dev/null 2>&1 && [ $count_api -lt $max_wait_api ]; do
    echo -n "."
    sleep 2
    count_api=$((count_api + 2))
done
echo ""

# Print final status check
echo "======================================================================="
echo "Status Check"
echo "======================================================================="

# Verify MLflow
echo -n "MLflow Service:   "
if curl -s http://localhost:5000/health > /dev/null 2>&1; then
    echo "RUNNING"
else
    echo "NOT RESPONDING"
fi

# Verify API
echo -n "API Service:      "
if curl -s http://localhost:8000/health/live > /dev/null 2>&1; then
    echo "RUNNING"
else
    echo "NOT RESPONDING"
fi

echo ""
echo "======================================================================="
echo "Services Initialized!"
echo "======================================================================="
echo ""
echo "API Live URL:  http://localhost:8000"
echo "API Docs:      http://localhost:8000/docs"
echo "MLflow UI:     http://localhost:5000"
echo ""
echo "Commands:"
echo "  View logs:   ./scripts/docker-logs.sh"
echo "  Stop server: ./scripts/docker-stop.sh"
echo ""