"""
Prediction endpoints.

Provides single and batch churn prediction.
"""

from fastapi import APIRouter, HTTPException, status
from typing import List

from src.api.models import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse
)
from src.api.services.predictor import prediction_service
from src.api.config import settings
from src.api.utils.logging import api_logger


router = APIRouter(
    prefix="/predict",
    tags=["Predictions"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Prediction failed"
        }
    }
)


@router.post(
    "/",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Churn (Single Customer)",
    description="Predict churn probability for a single customer.",
    response_description="Prediction result with churn probability and risk level"
)
async def predict_single(customer: CustomerData):
    """
    Predict churn for a single customer.
    
    **Input:** Customer data with all required features
    
    **Output:** Prediction with probability, risk level, and metadata
    
    **Example Request:**
```json
    {
      "customer_id": "CUST012345",
      "gender": "Male",
      "age": 45,
      "tenure": 24,
      "contract_type": "Month-to-Month",
      "internet_service": "Fiber",
      "monthly_charges": 85.50,
      "total_charges": 2051.20,
      "call_minutes": 450.0,
      "data_usage": 25.5,
      "complaints": 1,
      "recent_support_tickets": 0,
      "payment_method": "Credit Card",
      "late_payments": 0,
      "engagement": 1.2
    }
```
    
    **Example Response:**
```json
    {
      "customer_id": "CUST012345",
      "churn_prediction": "Yes",
      "churn_probability": 0.78,
      "risk_level": "High",
      "confidence": 0.78,
      "model_version": "v1",
      "prediction_date": "2026-04-19T10:30:00Z"
    }
```
    """
    api_logger.info(f"POST /predict - Customer: {customer.customer_id}")
    
    try:
        # Generate prediction
        result = prediction_service.predict_single(customer)
        
        return result
        
    except Exception as e:
        api_logger.error(f"Prediction failed for {customer.customer_id}: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "PredictionError",
                "message": f"Failed to generate prediction: {str(e)}",
                "customer_id": customer.customer_id
            }
        )


@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Churn (Batch)",
    description=f"Predict churn for multiple customers (max {settings.batch_size_limit}).",
    response_description="Batch prediction results with summary statistics"
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers in a single request.
    
    **Input:** List of customer data
    
    **Output:** List of predictions with summary statistics
    
    **Limits:**
    - Maximum customers per batch: {settings.batch_size_limit}
    - Predictions are processed in parallel
    
    **Example Request:**
```json
    {
      "customers": [
        {
          "customer_id": "CUST001",
          "gender": "Male",
          "age": 45,
          ...
        },
        {
          "customer_id": "CUST002",
          "gender": "Female",
          "age": 32,
          ...
        }
      ]
    }
```
    
    **Example Response:**
```json
    {
      "predictions": [
        {
          "customer_id": "CUST001",
          "churn_prediction": "Yes",
          "churn_probability": 0.78,
          ...
        }
      ],
      "total_predictions": 2,
      "high_risk_count": 1,
      "processing_time_seconds": 0.145
    }
```
    """
    api_logger.info(f"POST /predict/batch - Customers: {len(request.customers)}")
    
    # Validate batch size
    if len(request.customers) > settings.batch_size_limit:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "BatchSizeExceeded",
                "message": f"Batch size ({len(request.customers)}) exceeds limit ({settings.batch_size_limit})",
                "limit": settings.batch_size_limit
            }
        )
    
    try:
        # Generate predictions
        result = prediction_service.predict_batch(request.customers)
        
        return BatchPredictionResponse(**result)
        
    except Exception as e:
        api_logger.error(f"Batch prediction failed: {str(e)}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "BatchPredictionError",
                "message": f"Failed to generate batch predictions: {str(e)}",
                "batch_size": len(request.customers)
            }
        )


@router.get(
    "/model-info",
    status_code=status.HTTP_200_OK,
    summary="Get Model Information",
    description="Get metadata about the currently loaded model."
)
async def get_model_info():
    """
    Get information about the currently loaded model.
    
    Returns model name, version, stage, and load time.
    """
    from src.api.services.model_loader import model_loader
    
    metadata = model_loader.get_model_metadata()
    
    return {
        "model_name": metadata.get("name", "unknown"),
        "model_version": metadata.get("version", "unknown"),
        "model_stage": metadata.get("stage", "unknown"),
        "model_loaded": model_loader.is_model_loaded(),
        "feature_pipeline_loaded": model_loader.is_feature_pipeline_loaded(),
        "last_load_time": model_loader.get_load_time(),
        "threshold": settings.prediction_threshold
    }