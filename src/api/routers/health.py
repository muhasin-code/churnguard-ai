"""
Helath check endpoints.

Provides health status for monitoring and load balancers.
"""


from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any
import platform
import sys

from src.api.config import settings
from src.api.utils.logging import api_logger


router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Service unavailable"
        }
    }
)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    environment: str
    model_loaded: bool
    dependencies: Dict[str, str]


class ReadinessResponse(BaseModel):
    """Readiness check response model."""

    ready: bool
    checks: Dict[str, bool]
    message: str


@router.get(
    "/",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description="Basic health check endpoint. Returns API status and metadata."
)
async def health_check():
    """
    Basic health check.

    Returns:
        HealthResponse with API status
    
    Example response:
        {
            "status": "healthy",
            "timestamp": "2026-04-18T22:30:00",
            "version": "1.0.0",
            "environment": "development",
            "model_loaded": true,
            "dependencies": {
                "python": "3.10.12",
                "fastapi": "0.104.1",
                "mlflow": "2.8.0"
            }
        }
    """
    api_logger.info("Health check requested")

    import fastapi
    import mlflow

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.api_version,
        environment=settings.environment,
        model_loaded=True, # Will check actual model in Milestone 6.2
        dependencies={
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system(),
            "fastapi": fastapi.__version__,
            "mlflow": mlflow.__version__
        }
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    status_code=status.HTTP_200_OK,
    summary="Readiness Check",
    description="Readiness prob for Kubernetes/orchestrators. Checks if API is ready to serve traffic."
)
async def readiness_check():
    """
    Readiness check for orchestrators.

    Checks:
    - Configuration loaded
    - Model loaded (will implement in Milestone 6.2)
    - Dependencies avaliable

    Returns:
        ReadinessResponse indicating if service is ready
    """
    api_logger.debug("Readiness check requested")

    checks = {
        "config_loaded": True,
        "model_loaded": True, # Placeholder - will check actual model in 6.2
        "mlflow_accessible": True # Placeholder - will check connection in 6.2
    }

    all_ready = all(checks.values())

    return ReadinessResponse(
        ready=all_ready,
        checks=checks,
        message="All systems oprational" if all_ready else "Some systems not ready"
    )


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness Check",
    description="Liveness prob for Kubernetes. Returns 200 if API process is still alive."
)
async def liveness_check():
    """
    Liveness check - simplest health check

    Returns:
        {"alive": true}
    """
    return {"alive": True}