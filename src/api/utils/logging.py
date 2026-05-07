"""
Logging configuration for ChurnGuard API.

Provides structured logging with JSON formatting for production and 
human-readable formatting for development.
"""


import logging
import sys
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

from src.api.config import settings, get_project_root


def setup_logging(
    name: Optional[str] = None,
    log_level: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        name: Logger name (default to 'churnguard_api')
        log_level: Overrides log level from settings
    
    Returns:
        Configured logger instance
    """
    # Get logger
    logger_name = name or "churnguard_api"
    logger = logging.getLogger(logger_name)

    # Set level
    level = log_level or settings.log_level
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # =========================================================================
    # Console Handler
    # =========================================================================
    
    if settings.log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        if settings.log_format == "json":
            # JSON format for production
            json_formatter = jsonlogger.JsonFormatter(
                '%(asctime)s %(name)s %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(json_formatter)
        else:
            # Human-readable format for development
            console_formatter = logging.Formatter(
                '[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # =========================================================================
    # File Handler
    # =========================================================================
    
    if settings.log_to_file:
        # Ensure log directory exists
        log_file_path = get_project_root() / settings.log_file
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Always use JSON for file los (easier to parse)
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(json_formatter)

        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False

    return logger


# ============================================================================
# Global Logger Instance
# ============================================================================

# Create application logger
api_logger = setup_logging("churnguard_api")


# ============================================================================
# Request ID Context (for tracking requests)
# ============================================================================

import contextvars

# Context variable to store request ID
request_id_var = contextvars.ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_var.get()


def set_request_id(request_id: str):
    """Set request ID in context."""
    request_id_var.set(request_id)


class RequestIDFilter(logging.Filter):
    """Add request ID to log records."""

    def filter(self, record):
        record.request_id = get_request_id() or "no-request-id"
        return True


# Add request ID filter to logger
for handler in api_logger.handlers:
    handler.addFilter(RequestIDFilter())