"""
ChurnGuard AI - Prediction API

Main FastAPI application entry point.
"""


from fastapi import FastAPI, Request, responses, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import uuid

from src.api.config import settings
from src.api.routers import health
from src.api.utils.logging import api_logger, get_request_id, set_request_id
from src.api.exceptions import (
    ChurnGuardAPIException,
    ModelLoadError,
    PredictionError,
    ValidationError,
    BatchSizeExceededError,
    FeaturePreprocessingError
)


# ========================================================================
# Application Lifespan Events
# ========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Load model, validate config, initialize connections
    - Shutdown: Clean up resources, close connections
    """
    # Startup
    api_logger.info("=" * 70)
    api_logger.info("ChurnGuard AI - Starting API Server")
    api_logger.info("=" * 70)
    api_logger.info(f"Environment: {settings.environment}")
    api_logger.info(f"Debug Mode: {settings.debug}")
    api_logger.info(f"API Version: {settings.api_version}")
    api_logger.info(f"Model: {settings.model_name} ({settings.model_stage})")
    api_logger.info(f"MLflow URI: {settings.mlflow_tracking_uri}")

    # Load model and feature pipeline
    try:
        from src.api.services.model_loader import model_loader

        api_logger.info("Loading ML model and feature pipeline...")
        model = model_loader.get_model()
        pipeline = model_loader.get_feature_pipeline()

        api_logger.info("Model and pipeline loaded successfully")

        # Store in app state for access in endpoints
        app.state.model_loader = model_loader
    
    except Exception as e:
        api_logger.error(f"Failed to load model: {str(e)}")
        api_logger.error("   API will not be able to serve predictions")
        raise

    api_logger.info("Startup complete")

    yield

    # Shutdown
    api_logger.info("Shutting down API server...")
    # Clean up resources if needed
    api_logger.info("Shutdown complete")


# ========================================================================
# Create FastAPI Application
# ========================================================================

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None, # Disable in production
    redoc_url="/redoc" if settings.debug else None,
    debug=settings.debug
)

# ========================================================================
# Middleware Configuration
# ========================================================================

# CORS Middleware (allow cross-origin requests)
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    api_logger.info(f"CORS enabled for origins: {settings.cors_origins}")


# Request ID Middleware (for request tracing)
@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Add unique request ID to each request for tracing.

    The request ID is:
    - Generated as UUID4
    - Added to response headers
    - Available in logs via context variable
    """
    request_id = str(uuid.uuid4())
    set_request_id(request_id)

    # Log request
    api_logger.info(
        f"Request started",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown"
        }
    )

    # Process request
    start_time = time.time()

    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}"

    # Log response
    api_logger.info(
        f"Request completed",
        extra={
            "status_code": response.status_code,
            "duration_seconds": duration
        }
    )

    return response

# Request/Response Logging Middleware
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """
    Log all requests and responses for audit trail.
    
    Logs:
    - Request: method, path, client IP, body (for POST)
    - Response: status code, processing time
    """
    # Log request
    client_ip = request.client.host if request.client else "unknown"
    
    request_log = {
        "method": request.method,
        "path": request.url.path,
        "client_ip": client_ip,
        "user_agent": request.headers.get("user-agent", "unknown")
    }
    
    # Log request body for POST/PUT (but not for large batches)
    if request.method in ["POST", "PUT"]:
        # Read body (careful - can only read once!)
        body = await request.body()
        
        # Log body if small enough
        if len(body) < 5000:  # < 5KB
            try:
                import json
                body_json = json.loads(body)
                
                # Anonymize customer_id in logs (GDPR/privacy)
                if isinstance(body_json, dict) and 'customer_id' in body_json:
                    body_json['customer_id'] = body_json['customer_id'][:8] + "..."
                
                request_log['body_sample'] = body_json
            except:
                pass
    
    api_logger.info("API Request", extra=request_log)
    
    # Process request
    start_time = time.time()
    response = await call_next(request)
    processing_time = time.time() - start_time
    
    # Log response
    response_log = {
        "status_code": response.status_code,
        "processing_time_ms": round(processing_time * 1000, 2)
    }
    
    if response.status_code >= 400:
        api_logger.warning("API Response (Error)", extra=response_log)
    else:
        api_logger.info("API Response", extra=response_log)
    
    return response

# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle Pydantic validation errors.
    
    Returns 422 with detailed field-level error messages.
    """
    errors = exc.errors()
    
    # Format errors for better readability
    formatted_errors = []
    for error in errors:
        field = " -> ".join(str(loc) for loc in error['loc'])
        formatted_errors.append({
            "field": field,
            "message": error['msg'],
            "type": error['type'],
            "input": error.get('input', 'N/A')
        })
    
    api_logger.warning(
        f"Validation error on {request.url.path}",
        extra={
            "errors": formatted_errors,
            "error_count": len(formatted_errors)
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "errors": formatted_errors,
            "error_count": len(formatted_errors),
            "tip": "Check field types and value ranges"
        }
    )


@app.exception_handler(ChurnGuardAPIException)
async def churnguard_exception_handler(request: Request, exc: ChurnGuardAPIException):
    """
    Handle custom ChurnGuard exceptions.
    
    Returns appropriate status code with structured error info.
    """
    api_logger.error(
        f"{exc.__class__.__name__}: {exc.message}",
        extra={
            "status_code": exc.status_code,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler.
    
    Returns 500 Internal Server Error.
    """
    api_logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)
        }
    )
    
    # In production, hide internal error details
    error_message = str(exc) if settings.debug else "An unexpected error occurred"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": error_message,
            "request_id": get_request_id(),
            "tip": "Contact support if this persists"
        }
    )


# ========================================================================
# Include Routers
# ========================================================================

# Health check endpoints
app.include_router(health.router)

# Prediction endpoints
from src.api.routers import predict
app.include_router(predict.router)


# ========================================================================
# Root Endpoint
# ========================================================================


@app.get(
    "/",
    tags=["Root"],
    summary="API Root",
    description="Welcome endpoint with API information and links"
)
async def root():
    """
    API root endpoint.

    Returns:
        Welcome message with API metadata and links
    """
    return {
        "message": "Welcome to ChurnGuard AI Prediction API",
        "version": settings.api_version,
        "environment": settings.environment,
        "documentation": "/docs" if settings.debug else "Contact administator",
        "health": "/health",
        "endpoints": {
            "health": "/health",
            "readiness": "/health/ready",
            "liveness": "/health/live",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "model_info": "/predict/model-info"
        }
    }


# ========================================================================
# Utility Function Imports for Exception Handler
# ========================================================================

from src.api.utils.logging import get_request_id