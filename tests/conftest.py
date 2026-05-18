"""
Pytest configuration and shared fixtures.

This file is automatically loaded by pytest and provides
fixtures that can be used across all tests.
"""

import pytest
from fastapi.testclient import TestClient
from faker import Faker
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app
from src.api.models import CustomerData


# ============================================================================
# Test Client Fixture
# ============================================================================

@pytest.fixture(scope="module")
def client():
    """
    FastAPI test client.
    
    Scope: module (created once per test module for speed)
    """
    with TestClient(app) as test_client:
        yield test_client


# ============================================================================
# Faker Fixture
# ============================================================================

@pytest.fixture(scope="session")
def fake():
    """Faker instance for generating test data."""
    return Faker()


# ============================================================================
# Sample Customer Data Fixtures
# ============================================================================

@pytest.fixture
def valid_customer_data():
    """Valid customer data for testing predictions."""
    return {
        "customer_id": "TEST_001",
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


@pytest.fixture
def high_risk_customer_data():
    """Customer data likely to churn (high risk)."""
    return {
        "customer_id": "TEST_HIGH_RISK",
        "gender": "Male",
        "age": 45,
        "tenure": 6,  # Short tenure
        "contract_type": "Month-to-Month",  # No commitment
        "internet_service": "Fiber",
        "monthly_charges": 95.50,  # High charges
        "total_charges": 573.00,
        "call_minutes": 450.0,
        "data_usage": 25.5,
        "complaints": 3,  # Many complaints
        "recent_support_tickets": 2,
        "payment_method": "Credit Card",
        "late_payments": 2,  # Late payments
        "engagement": 0.6  # Low engagement
    }


@pytest.fixture
def low_risk_customer_data():
    """Customer data unlikely to churn (low risk)."""
    return {
        "customer_id": "TEST_LOW_RISK",
        "gender": "Female",
        "age": 35,
        "tenure": 60,  # Long tenure
        "contract_type": "Two Year",  # Strong commitment
        "internet_service": "Fiber",
        "monthly_charges": 50.00,  # Reasonable charges
        "total_charges": 3000.00,
        "call_minutes": 400.0,
        "data_usage": 20.0,
        "complaints": 0,  # No complaints
        "recent_support_tickets": 0,
        "payment_method": "UPI",
        "late_payments": 0,
        "engagement": 1.8  # High engagement
    }


@pytest.fixture
def invalid_customer_data_age():
    """Invalid customer data (age < 18)."""
    return {
        "customer_id": "TEST_INVALID_AGE",
        "gender": "Male",
        "age": 16,  # Invalid: under 18
        "tenure": 12,
        "contract_type": "One Year",
        "internet_service": "DSL",
        "monthly_charges": 55.20,
        "total_charges": 662.40,
        "call_minutes": 320.0,
        "data_usage": 15.0,
        "complaints": 0,
        "recent_support_tickets": 0,
        "payment_method": "UPI",
        "late_payments": 0,
        "engagement": 1.3
    }


@pytest.fixture
def invalid_customer_data_total_charges():
    """Invalid customer data (inconsistent total_charges)."""
    return {
        "customer_id": "TEST_INVALID_TOTAL",
        "gender": "Male",
        "age": 45,
        "tenure": 60,
        "contract_type": "Two Year",
        "internet_service": "Fiber",
        "monthly_charges": 85.50,
        "total_charges": 500.00,  # Invalid: too low for tenure
        "call_minutes": 450.0,
        "data_usage": 25.5,
        "complaints": 1,
        "recent_support_tickets": 0,
        "payment_method": "Credit Card",
        "late_payments": 0,
        "engagement": 1.2
    }


@pytest.fixture
def batch_customer_data(valid_customer_data, high_risk_customer_data, low_risk_customer_data):
    """Batch of customer data for batch prediction testing."""
    return {
        "customers": [
            valid_customer_data,
            high_risk_customer_data,
            low_risk_customer_data
        ]
    }


# ============================================================================
# Helper Functions
# ============================================================================

@pytest.fixture
def generate_random_customer(fake):
    """
    Factory fixture to generate random customer data.
    
    Usage:
        customer = generate_random_customer()
    """
    def _generate():
        tenure = fake.random_int(min=0, max=72)
        monthly_charges = round(fake.random.uniform(20.0, 120.0), 2)
        internet_service = fake.random_element(["DSL", "Fiber", "No Service"])
        data_usage = 0.0 if internet_service == "No Service" else round(fake.random.uniform(5.0, 50.0), 1)
        
        return {
            "customer_id": f"TEST_{fake.uuid4()[:8].upper()}",
            "gender": fake.random_element(["Male", "Female"]),
            "age": fake.random_int(min=18, max=80),
            "tenure": tenure,
            "contract_type": fake.random_element(["Month-to-Month", "One Year", "Two Year"]),
            "internet_service": internet_service,
            "monthly_charges": monthly_charges,
            "total_charges": round(tenure * monthly_charges * fake.random.uniform(0.8, 1.2), 2),
            "call_minutes": round(fake.random.uniform(100.0, 800.0), 1),
            "data_usage": data_usage,
            "complaints": fake.random_int(min=0, max=5),
            "recent_support_tickets": fake.random_int(min=0, max=3),
            "payment_method": fake.random_element(["Credit Card", "Debit Card", "UPI", "Cash"]),
            "late_payments": fake.random_int(min=0, max=3),
            "engagement": round(fake.random.uniform(0.2, 1.9), 2)
        }
    
    return _generate