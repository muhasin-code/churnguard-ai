"""
Pydantic models for API request/response validation.

These models define the contract between API clients and the server.
FastAPI uses these for automatic validation, serialization, and document generation.
"""


from pydantic import BaseModel, Field, validator, root_validator
from typing import Optional, List
from datetime import datetime
from enum import Enum


# ==========================================================================
# Enums for Categorical Fields
# ==========================================================================

class ContractType(str, Enum):
    """Contract type options."""
    MONTH_TO_MONTH = "Month-to-Month"
    ONE_YEAR = "One Year"
    TWO_YEAR = "Two Year"


class InternetService(str, Enum):
    """Internet service options"""
    DSL = "DSL"
    FIBER = "Fiber"
    NO_SERVICE = "No Service"


class PaymentMethod(str, Enum):
    """Payment method options."""
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    UPI = "UPI"
    CASH = "Cash"


class Gender(str, Enum):
    """Gender options."""
    MALE = "Male"
    Female = "Female"


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# ==========================================================================
# Request Models
# ==========================================================================

class CustomerData(BaseModel):
    """
    Customer data for churn prediction.

    This model represents the input features required for prediction.
    All fields are validated automatically by FastAPI/Pydantic.
    """
    # Identifiers
    customer_id: str = Field(
        ...,
        description="Unique customer identifier",
        example="CUST012345",
        min_length=1,
        max_length=50
    )

    # Demographics
    gender: Gender = Field(
        ...,
        description="Customer gender",
        example="Male"
    )

    age: int = Field(
        ...,
        description="Customer age",
        example=45,
        ge=18, # Greater than or equel to 18
        le=100, # Less than or equel to 100
    )

    # Account Information
    tenure: int = Field(
        ...,
        description="Number of months the customer has been with the company",
        example=24,
        ge=0,
        le=100
    )

    contract_type: ContractType = Field(
        ...,
        description="Type of contract",
        example="Month-to-Month"
    )

    # Services
    internet_service: InternetService = Field(
        ...,
        description="Type of internet service",
        example="Fiber"
    )

    # Billing
    monthly_charges: float = Field(
        ...,
        description="Monthly charges in dollars",
        example=85.50,
        ge=0.0, # Greater than 0
        le=200.0
    )

    total_charges: float = Field(
        ...,
        description="Total charges to date in dollars",
        example=2051.20,
        ge=0.0,
        le=10000.0
    )

    # Usage
    call_minutes: float = Field(
        ...,
        description="Average monthly call minutes",
        example=450.0,
        ge=0.0,
        le=10000.0
    )

    data_usage: float = Field(
        ...,
        description="Average monthly data usage in GB",
        example=25.5,
        ge=0.0,
        le=1000.0
    )

    # Customer Service
    complaints: int = Field(
        ...,
        description="Number of complaints files",
        example=1,
        ge=0,
        le=10
    )

    recent_support_tickets: int = Field(
        ...,
        description="Number of support tickets in last 3 months",
        example=0,
        ge=0,
        le=10
    )

    # Payment
    payment_method: PaymentMethod = Field(
        ...,
        description="Payment method",
        example="Credit Card"
    )

    late_payments: int = Field(
        ...,
        description="Number of late payments",
        example=0,
        ge=0,
        le=10
    )

    # Engagement
    engagement: float = Field(
        ...,
        description="Engagement score (0-2 scale)",
        example=1.2,
        ge=0.0,
        le=2.0
    )

    # =========================================================================
    # VALIDATORS - Business Logic & Data Quality
    # =========================================================================
    
    @validator('age')
    def validate_age_realistic(cls, v):
        """Validate age is in realistic range."""
        if v < 18:
            raise ValueError("Customer must be at least 18 years old")
        if v > 100:
            raise ValueError("Age seems unrealistic (>100)")
        return v
    
    @validator('tenure')
    def validate_tenure_realistic(cls, v):
        """Validate tenure is realistic."""
        if v < 0:
            raise ValueError("Tenure cannot be negative")
        if v > 100:
            raise ValueError("Tenure exceeds 100 months (8+ years) - seems unrealistic")
        return v
    
    @validator('monthly_charges')
    def validate_monthly_charges(cls, v):
        """Validate monthly charges are in realistic range."""
        if v <= 0:
            raise ValueError("Monthly charges must be positive")
        if v > 200:
            raise ValueError("Monthly charges exceed $200 - seems unrealistic for telecom")
        return v
    
    @validator('total_charges')
    def validate_total_charges(cls, v, values):
        """
        Validate total_charges is consistent with tenure and monthly_charges.
        
        Expected: total_charges ≈ tenure × monthly_charges
        Allow 50-150% variance for promotions/discounts.
        """
        if 'tenure' in values and 'monthly_charges' in values:
            tenure = values['tenure']
            monthly = values['monthly_charges']
            
            if tenure == 0:
                # New customer - total charges should be near zero
                if v > monthly * 2:
                    raise ValueError(
                        f"New customer (tenure=0) but total_charges=${v:.2f} is too high. "
                        f"Expected < ${monthly * 2:.2f}"
                    )
            else:
                # Existing customer - check consistency
                expected_min = tenure * monthly * 0.5
                expected_max = tenure * monthly * 1.5
                
                if v < expected_min:
                    raise ValueError(
                        f"total_charges=${v:.2f} is too low. "
                        f"Expected ${expected_min:.2f} - ${expected_max:.2f} "
                        f"(tenure={tenure} months × monthly_charges=${monthly:.2f})"
                    )
                
                if v > expected_max:
                    raise ValueError(
                        f"total_charges=${v:.2f} is too high. "
                        f"Expected ${expected_min:.2f} - ${expected_max:.2f} "
                        f"(tenure={tenure} months × monthly_charges=${monthly:.2f})"
                    )
        
        return v
    
    @validator('call_minutes')
    def validate_call_minutes(cls, v):
        """Validate call minutes are realistic."""
        if v < 0:
            raise ValueError("Call minutes cannot be negative")
        if v > 10000:
            raise ValueError("Call minutes exceed 10,000/month - seems unrealistic")
        return v
    
    @validator('data_usage')
    def validate_data_usage(cls, v):
        """Validate data usage is realistic."""
        if v < 0:
            raise ValueError("Data usage cannot be negative")
        if v > 1000:
            raise ValueError("Data usage exceeds 1TB/month - seems unrealistic for consumer")
        return v
    
    @validator('engagement')
    def validate_engagement_range(cls, v):
        """Validate engagement score is in expected range."""
        if v < 0:
            raise ValueError("Engagement score cannot be negative")
        if v > 2:
            raise ValueError("Engagement score exceeds maximum (2.0)")
        return v
    
    @validator('complaints', 'late_payments')
    def validate_non_negative_counts(cls, v, field):
        """Validate count fields are non-negative."""
        if v < 0:
            raise ValueError(f"{field.name} cannot be negative")
        return v
    
    @root_validator
    def validate_cross_field_consistency(cls, values):
        """
        Validate cross-field business logic.
        
        Checks:
        1. Contract type vs tenure alignment
        2. High charges with low engagement (churn indicator)
        3. Service usage consistency
        """
        # Extract values
        tenure = values.get('tenure', 0)
        contract_type = values.get('contract_type', '')
        monthly_charges = values.get('monthly_charges', 0)
        engagement = values.get('engagement', 0)
        internet_service = values.get('internet_service', '')
        data_usage = values.get('data_usage', 0)
        
        # Check 1: Contract type vs tenure alignment
        if contract_type == 'Two Year' and tenure < 3:
            # Warning: Two-year contract but very new customer
            # Don't raise error - this is possible (new signup)
            pass
        
        if contract_type == 'Month-to-Month' and tenure > 60:
            # Unusual: 5+ years but still month-to-month
            # Don't raise error - customer choice
            pass
        
        # Check 2: High charges with low engagement
        if monthly_charges > 80 and engagement < 0.5:
            # Warning: Paying a lot but barely using service
            # Don't raise error - this is a legitimate churn signal
            pass
        
        # Check 3: Service usage consistency
        if internet_service == 'No Service' and data_usage > 0:
            raise ValueError(
                f"internet_service='No Service' but data_usage={data_usage} GB. "
                f"Data usage should be 0 if no internet service."
            )
        
        # Check 4: Engagement calculation consistency
        # Engagement is derived from CallMinutes and DataUsage
        # Basic sanity check: if both are 0, engagement should be low
        call_minutes = values.get('call_minutes', 0)
        if call_minutes == 0 and data_usage == 0 and engagement > 1.5:
            # Warning: No usage but high engagement score
            # Don't raise error - engagement might be calculated differently
            pass
        
        return values
    
    class Config:
        """Pydantic model configuration."""
        use_enum_values = True # Use enum values instead of names
        schema_extra = {
            "example": {
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
        }


class BatchPredictionRequest(BaseModel):
    """
    Batch prediction request.

    Allows predicting multiple customers in a single API call.
    """
    
    customers: List[CustomerData] = Field(
        ...,
        description="List of customers to predict",
        min_items=1
    )

    class Config:
        schema_extra = {
            "example": {
                "customers": [
                    {
                        "customer_id": "CUST001",
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
                    },
                    {
                        "customer_id": "CUST002",
                        "gender": "Female",
                        "age": 32,
                        "tenure": 48,
                        "contract_type": "Two Year",
                        "internet_service": "DSL",
                        "monthly_charges": 55.20,
                        "total_charges": 2649.60,
                        "call_minutes": 320.0,
                        "data_usage": 15.0,
                        "complaints": 0,
                        "recent_support_tickets": 0,
                        "payment_method": "UPI",
                        "late_payments": 0,
                        "engagement": 1.5
                    }
                ]
            }
        }


# ============================================================================
# Response Models
# ============================================================================

class PredictionResponse(BaseModel):
    """
    Single prediction response.

    Contains the prediction result with metadata.
    """

    customer_id: str = Field(
        ...,
        description="Customer identifier from request"
    )

    churn_prediction: str = Field(
        ...,
        description="Binary prediction: 'Yes' or 'No'",
        example="Yes"
    )

    churn_probability: float = Field(
        ...,
        description="Probability of churn (0.0 to 1.0)",
        example=0.78,
        ge=0.0,
        le=1.0
    )

    risk_level: RiskLevel = Field(
        ...,
        description="Risk categorization",
        example="High"
    )
    
    confidence: float = Field(
        ...,
        description="Model confidence (0.0 to 1.0)",
        example=0.78,
        ge=0.0,
        le=1.0
    )

    model_version: str = Field(
        ...,
        description="Model version used for prediction",
        example="v1"
    )
    
    prediction_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of prediction"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST012345",
                "churn_prediction": "Yes",
                "churn_probability": 0.78,
                "risk_level": "High",
                "confidence": 0.78,
                "model_version": "v1",
                "prediction_date": "2026-04-19T10:30:00Z"
            }
        }


class BatchPredictionResponse(BaseModel):
    """
    Batch prediction response.
    
    Contains predictions for all customers in the batch.
    """
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )
    
    total_predictions: int = Field(
        ...,
        description="Total number of predictions made"
    )
    
    high_risk_count: int = Field(
        ...,
        description="Number of high/critical risk customers"
    )
    
    processing_time_seconds: float = Field(
        ...,
        description="Time taken to process batch"
    )

    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "customer_id": "CUST001",
                        "churn_prediction": "Yes",
                        "churn_probability": 0.78,
                        "risk_level": "High",
                        "confidence": 0.78,
                        "model_version": "v1",
                        "prediction_date": "2026-04-19T10:30:00Z"
                    }
                ],
                "total_predictions": 1,
                "high_risk_count": 1,
                "processing_time_seconds": 0.145
            }
        }


class ErrorResponse(BaseModel):
    """
    Error response model.
    
    Returned when prediction fails.
    """
    
    error: str = Field(
        ...,
        description="Error type",
        example="PredictionError"
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message",
        example="Failed to load model"
    )
    
    customer_id: Optional[str] = Field(
        None,
        description="Customer ID if applicable"
    )
    
    details: Optional[dict] = Field(
        None,
        description="Additional error details"
    )

    class Config:
        schema_extra = {
            "example": {
                "error": "PredictionError",
                "message": "Failed to load model from MLflow",
                "customer_id": "CUST012345",
                "details": {
                    "model_name": "churnguard-classifier",
                    "stage": "Production"
                }
            }
        }