"""
Custom exceptions for ChurnGuard API.

Provides specific exception types for different error scenarios
with structured error information.
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class ChurnGuardAPIException(Exception):
    """Base exception for all ChurnGuard API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API response."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ModelLoadError(ChurnGuardAPIException):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details
        )


class PredictionError(ChurnGuardAPIException):
    """Raised when prediction generation fails."""
    
    def __init__(
        self,
        message: str,
        customer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if customer_id:
            details['customer_id'] = customer_id
        
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )


class ValidationError(ChurnGuardAPIException):
    """Raised when data validation fails."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if field:
            details['field'] = field
        if invalid_value is not None:
            details['invalid_value'] = str(invalid_value)
        
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class BatchSizeExceededError(ChurnGuardAPIException):
    """Raised when batch size exceeds limit."""
    
    def __init__(self, batch_size: int, limit: int):
        super().__init__(
            message=f"Batch size ({batch_size}) exceeds limit ({limit})",
            status_code=status.HTTP_400_BAD_REQUEST,
            details={
                "batch_size": batch_size,
                "limit": limit
            }
        )


class FeaturePreprocessingError(ChurnGuardAPIException):
    """Raised when feature preprocessing fails."""
    
    def __init__(
        self,
        message: str,
        customer_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if customer_id:
            details['customer_id'] = customer_id
        
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details
        )