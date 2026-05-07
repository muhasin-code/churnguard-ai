"""
Configuration Management for ChurnGuard API.

Loads settings from environment variables using Pydantic v1 BaseSettings.
Provides type-safe access to configuration across the application.
"""


from pydantic import BaseSettings, Field, validator
from typing import List, Optional
import os
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Example:
        from src.api.config import settings
        print(settings.mlflow_tracking_uri)
        print(settings.api_port)
    """

    # =========================================================================
    # MLflow Configuration
    # =========================================================================

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )

    mlflow_registry_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow model registry URI"
    )

    model_name: str = Field(
        default="churnguard-classifier",
        description="Registered model registry URI"
    )

    model_stage: str = Field(
        default="Production",
        description="Model stage to load (None, Staging, Production, Archived)"
    )

    # =========================================================================
    # PI Server Configuration
    # =========================================================================

    api_title: str = Field(
        default="ChurnGuard AI - Prediction API",
        description="API title shown in documentation"
    )
    
    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )

    api_description: str = Field(
        default="Production ML API for customer churn prediction",
        description="API description shown in documentation"
    )

    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind the API server"
    )

    api_port: int = Field(
        default=8000,
        description="Port to run the API server"
    )

    api_workers: int = Field(
        default=1,
        description="Number of worker processes"
    )

    api_reload: bool = Field(
        default=True,
        description="Auto-reload on code changes (development only)"
    )

    # =========================================================================
    # Model Inference Configuration
    # =========================================================================

    prediction_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold for churn prediction"
    )

    batch_size_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum number of predictions per batch request"
    )

    enable_probability_output: bool = Field(
        default=True,
        description="Include probability scores in response"
    )

    enable_feature_importance: bool = Field(
        default=False,
        description="Include SHAP feature importance in response (expensive)"
    )

    # =========================================================================
    # Feature Engineering
    # =========================================================================

    feature_pipeline_path: str = Field(
        default="models/feature_pipeline.pkl",
        description="Path to feature engineering pipeline"
    )

    # =========================================================================
    # Logging Configuration
    # =========================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    log_file: str = Field(
        default="logs/api.log",
        description="Path to log file"
    )

    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'text'"
    )

    log_to_console: bool = Field(
        default=True,
        description="Enable console logging"
    )

    log_to_file: bool = Field(
        default=True,
        description="Enable file logging"
    )

    # =========================================================================
    # Environment
    # =========================================================================

    environment: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )

    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )

    # =========================================================================
    # Security
    # =========================================================================

    enable_api_key: bool = Field(
        default=False,
        description="Require API key for requests"
    )

    cors_enabled: bool = Field(
        default=True,
        description="Enable CORS"
    )

    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )

    # =========================================================================
    # Performance & Limits
    # =========================================================================

    request_timeout_seconds: int = Field(
        default=30,
        description="Request timeout seconds"
    )

    max_concurrent_requests: int = Field(
        default=100,
        description="Maximum concurrent requests"
    )

    enable_response_cache: bool = Field(
        default=False,
        description="Enable response caching"
    )

    # =========================================================================
    # Validators
    # =========================================================================

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level is valid."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment is valid."""
        valid_envs = ['development', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"environment must be one of {valid_envs}")
        return v.lower()
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            # Handle string like: "['http://localhost:3000', 'http://localhost:8000']"
            import ast
            try:
                return ast.literal_eval(v)
            except:
                # Handle comma-separated string: 'http://localhost:3000', 'http://localhost:8000'
                return [origin.strip() for origin in v.split(',')]
        return v
    
    # =========================================================================
    # Pydantic v1 Configuration
    # =========================================================================
    
    class Config:
        """Pydantic v1 configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False # Environment variables are case-insensitive

        # In Pydantiv v1, use 'fields' instead of 'extra'
        # Allow extra fields in .env without errors
        extra = "ignore"


# =========================================================================
# Global Settings Instance
# =========================================================================

# Create singleton instance
# This will automatically load from .env file
settings = Settings()

# =========================================================================
# Helper Functions
# =========================================================================

def get_project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


def ensure_directories():
    """Create required directories if they don't exist."""
    root = get_project_root()

    directories = [
        root / "logs",
        root / "models",
        root / "data" / "processed"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_settings():
    """
    Validate settings and check dependencies.

    Raises:
        ValueError: If critical settings are invalid
        FileNotFOundError: If required file are missing
    """
    root = get_project_root()

    # Check feature pipeline exists
    pipeline_path = root / settings.feature_pipeline_path
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"Feature pipeline not found: {pipeline_path}\n"
            f"Run feature engineering first: python scripts/engineer_features.py"
        )
    
    # Validate threshold
    if not (0.0 <= settings.prediction_threshold <= 1.0):
        raise ValueError(
            f"prediction_threshold must be between 0 and 1, "
            f"got {settings.prediction_threshold}"
        )
    
    print("Settings validated successfully")

# ============================================================================
# Initialization
# ============================================================================

# Ensure directories exist
ensure_directories()

# Validate settings on module import
try:
    validate_settings()
except Exception as e:
    print(f"Settings validation warning: {e}")
    print("   API may not function correctly")