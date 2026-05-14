"""
Prediction service.

Handles feature preprocessing and prediction logic.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from enum import Enum
import time

from src.api.models import (
    CustomerData,
    PredictionResponse,
    RiskLevel
)
from src.api.services.model_loader import model_loader
from src.api.config import settings
from src.api.utils.logging import api_logger


class PredictionService:
    """
    Handles customer churn predictions.
    
    Workflow:
    1. Convert raw customer data to DataFrame
    2. Preprocess features using feature pipeline
    3. Generate predictions using ML model
    4. Format and return response
    """
    
    def __init__(self):
        """Initialize prediction service."""
        self.model = None
        self.feature_pipeline = None
        self._ensure_loaded()
    
    def _ensure_loaded(self):
        """Ensure model and pipeline are loaded."""
        if self.model is None:
            api_logger.info("Loading model in prediction service...")
            self.model = model_loader.get_model()
        
        if self.feature_pipeline is None:
            api_logger.info("Loading feature pipeline in prediction service...")
            self.feature_pipeline = model_loader.get_feature_pipeline()
    
    # =========================================================================
    # Data Conversion
    # =========================================================================
    
    def customer_to_dataframe(self, customer: CustomerData) -> tuple:
        """
        Convert a single CustomerData object to a DataFrame row.
        
        Maps snake_case Pydantic field names to PascalCase column names
        expected by the feature engineering pipeline.
        
        Args:
            customer: Validated customer data
            
        Returns:
            Tuple of (DataFrame with one row, customer_id string)
        """
        customer_id = customer.customer_id

        def _plain(v: Any) -> Any:
            """Sklearn / pandas expect primitive labels, not str-Enum instances."""
            return v.value if isinstance(v, Enum) else v
        
        # Map Pydantic fields (snake_case) -> Pipeline columns (PascalCase)
        data = {
            "CustomerID":            [customer.customer_id],
            "Gender":                [_plain(customer.gender)],
            "Age":                   [customer.age],
            "Tenure":                [customer.tenure],
            "ContractType":          [_plain(customer.contract_type)],
            "InternetService":       [_plain(customer.internet_service)],
            "MonthlyCharges":        [customer.monthly_charges],
            "TotalCharges":          [customer.total_charges],
            "CallMinutes":           [customer.call_minutes],
            "DataUsage":             [customer.data_usage],
            "Complaints":            [customer.complaints],
            "RecentSupportTickets":  [customer.recent_support_tickets],
            "PaymentMethod":         [_plain(customer.payment_method)],
            "LatePayments":          [customer.late_payments],
            "Engagement":            [customer.engagement],
        }
        
        df = pd.DataFrame(data)
        
        api_logger.debug(f"Converted customer {customer_id} to DataFrame: {df.shape}")
        
        return df, customer_id
    
    def batch_to_dataframe(self, customers: List[CustomerData]) -> pd.DataFrame:
        """
        Convert batch of customers to DataFrame with proper categorical encoding.
        
        Args:
            customers: List of validated customer data
            
        Returns:
            DataFrame with multiple rows, list of customer IDs
        """
        customer_ids = []
        rows = []
        
        for customer in customers:
            df, customer_id = self.customer_to_dataframe(customer)
            rows.append(df.iloc[0])
            customer_ids.append(customer_id)
        
        batch_df = pd.DataFrame(rows)
        
        # Reset index to avoid issues
        batch_df = batch_df.reset_index(drop=True)
        
        api_logger.debug(f"Converted {len(customers)} customers to DataFrame: {batch_df.shape}")
        
        return batch_df, customer_ids
    
    # =========================================================================
    # Feature Engineering
    # =========================================================================
    
    def preprocess_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Raw feature DataFrame
            
        Returns:
            Preprocessed feature array
        """
        api_logger.debug("Applying feature engineering pipeline...")
        
        try:
            # Transform using feature pipeline
            X = self.feature_pipeline.transform(df)
            
            api_logger.debug(f"Features preprocessed: {X.shape}")
            
            return X
            
        except Exception as e:
            api_logger.error(f"Feature preprocessing failed: {str(e)}")
            raise RuntimeError(f"Feature preprocessing failed: {str(e)}")
    
    # =========================================================================
    # Prediction
    # =========================================================================
    
    def predict_single(self, customer: CustomerData) -> PredictionResponse:
        """
        Predict churn for a single customer.
        
        Args:
            customer: Validated customer data
            
        Returns:
            Prediction response
        """
        api_logger.info(f"Predicting churn for customer: {customer.customer_id}")
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df, customer_id = self.customer_to_dataframe(customer)
            
            # Preprocess features
            X = self.preprocess_features(df)
            
            # Generate prediction
            prediction = self.model.predict(X)[0]  # 0 or 1
            probability = self.model.predict_proba(X)[0, 1]  # Probability of churn (class 1)
            
            # Classify risk level
            risk_level = self._classify_risk(probability)
            
            # Get model metadata
            model_metadata = model_loader.get_model_metadata()
            
            # Create response
            response = PredictionResponse(
                customer_id=customer_id,
                churn_prediction="Yes" if prediction == 1 else "No",
                churn_probability=round(float(probability), 4),
                risk_level=risk_level,
                confidence=round(float(probability) if prediction == 1 else 1 - float(probability), 4),
                model_version=f"v{model_metadata.get('version', 'unknown')}"
            )
            
            prediction_time = time.time() - start_time
            
            api_logger.info(
                f"Prediction complete for {customer_id}: "
                f"{response.churn_prediction} ({response.churn_probability:.4f}) "
                f"in {prediction_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            api_logger.error(f"Prediction failed for {customer.customer_id}: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def predict_batch(self, customers: List[CustomerData]) -> Dict[str, Any]:
        """
        Predict churn for multiple customers.
        
        Args:
            customers: List of validated customer data
            
        Returns:
            Dict with predictions and metadata
        """
        api_logger.info(f"Batch prediction for {len(customers)} customers")
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df, customer_ids = self.batch_to_dataframe(customers)
            
            # Preprocess features
            X = self.preprocess_features(df)
            
            # Generate predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            
            # Get model metadata
            model_metadata = model_loader.get_model_metadata()
            
            # Create responses
            responses = []
            high_risk_count = 0
            
            for i, (customer_id, pred, prob) in enumerate(zip(customer_ids, predictions, probabilities)):
                risk_level = self._classify_risk(prob)
                
                if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    high_risk_count += 1
                
                response = PredictionResponse(
                    customer_id=customer_id,
                    churn_prediction="Yes" if pred == 1 else "No",
                    churn_probability=round(float(prob), 4),
                    risk_level=risk_level,
                    confidence=round(float(prob) if pred == 1 else 1 - float(prob), 4),
                    model_version=f"v{model_metadata.get('version', 'unknown')}"
                )
                
                responses.append(response)
            
            processing_time = time.time() - start_time
            
            api_logger.info(
                f"Batch prediction complete: {len(responses)} predictions "
                f"in {processing_time:.3f}s ({high_risk_count} high-risk)"
            )
            
            return {
                "predictions": responses,
                "total_predictions": len(responses),
                "high_risk_count": high_risk_count,
                "processing_time_seconds": round(processing_time, 3)
            }
            
        except Exception as e:
            api_logger.error(f"Batch prediction failed: {str(e)}")
            raise RuntimeError(f"Batch prediction failed: {str(e)}")
    
    # =========================================================================
    # Risk Classification
    # =========================================================================
    
    def _classify_risk(self, probability: float) -> RiskLevel:
        """
        Classify customer risk based on churn probability.
        
        Thresholds:
        - Critical: >= 0.80
        - High: 0.65 - 0.79
        - Medium: 0.45 - 0.64
        - Low: < 0.45
        
        Args:
            probability: Churn probability (0-1)
            
        Returns:
            Risk level enum
        """
        if probability >= 0.80:
            return RiskLevel.CRITICAL
        elif probability >= 0.65:
            return RiskLevel.HIGH
        elif probability >= 0.45:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


# ============================================================================
# Global Instance
# ============================================================================

# Create global prediction service
prediction_service = PredictionService()