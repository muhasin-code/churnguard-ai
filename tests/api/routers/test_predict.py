"""
Tests for prediction endpoints.

Tests:
- POST /predict/ - Single prediction
- POST /predict/batch - Batch predictions
- GET /predict/model-info - Model information
- Validation errors
- Edge cases
"""

import pytest


class TestPredictSingleEndpoint:
    """Test suite for single prediction endpoint."""
    
    def test_predict_valid_customer(self, client, valid_customer_data):
        """Test prediction with valid customer data."""
        response = client.post("/predict/", json=valid_customer_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["customer_id"] == valid_customer_data["customer_id"]
        assert data["churn_prediction"] in ["Yes", "No"]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["Low", "Medium", "High", "Critical"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert "model_version" in data
        assert "prediction_date" in data
    
    def test_predict_high_risk_customer(self, client, high_risk_customer_data):
        """Test prediction for high-risk customer."""
        response = client.post("/predict/", json=high_risk_customer_data)
        
        assert response.status_code == 200
        
        data = response.json()
        # High risk customer should have high churn probability
        assert data["churn_probability"] > 0.6
        assert data["risk_level"] in ["High", "Critical"]
    
    def test_predict_low_risk_customer(self, client, low_risk_customer_data):
        """Test prediction for low-risk customer."""
        response = client.post("/predict/", json=low_risk_customer_data)
        
        assert response.status_code == 200
        
        data = response.json()
        # Low risk customer should have low churn probability
        assert data["churn_probability"] < 0.5
        assert data["risk_level"] in ["Low", "Medium"]
    
    def test_predict_invalid_age(self, client, invalid_customer_data_age):
        """Test prediction with invalid age (< 18)."""
        response = client.post("/predict/", json=invalid_customer_data_age)
        
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert data["error"] == "ValidationError"
        assert "errors" in data
        
        # Check that age validation error is present
        errors = data["errors"]
        age_errors = [e for e in errors if "age" in e["field"].lower()]
        assert len(age_errors) > 0
    
    def test_predict_invalid_total_charges(self, client, invalid_customer_data_total_charges):
        """Test prediction with inconsistent total_charges."""
        response = client.post("/predict/", json=invalid_customer_data_total_charges)
        
        assert response.status_code == 422  # Validation error
        
        data = response.json()
        assert data["error"] == "ValidationError"
        
        # Check error message mentions total_charges
        errors = data["errors"]
        total_charge_errors = [e for e in errors if "total_charges" in e["field"].lower()]
        assert len(total_charge_errors) > 0
    
    def test_predict_missing_field(self, client, valid_customer_data):
        """Test prediction with missing required field."""
        # Remove required field
        incomplete_data = valid_customer_data.copy()
        del incomplete_data["age"]
        
        response = client.post("/predict/", json=incomplete_data)
        
        assert response.status_code == 422
    
    def test_predict_wrong_data_type(self, client, valid_customer_data):
        """Test prediction with wrong data type."""
        invalid_data = valid_customer_data.copy()
        invalid_data["age"] = "thirty-two"  # Should be int
        
        response = client.post("/predict/", json=invalid_data)
        
        assert response.status_code == 422
    
    def test_predict_invalid_enum_value(self, client, valid_customer_data):
        """Test prediction with invalid enum value."""
        invalid_data = valid_customer_data.copy()
        invalid_data["contract_type"] = "Invalid Contract"
        
        response = client.post("/predict/", json=invalid_data)
        
        assert response.status_code == 422
    
    @pytest.mark.parametrize("field,value", [
        ("age", -5),
        ("tenure", -10),
        ("monthly_charges", -50.00),
        ("total_charges", -100.00),
        ("call_minutes", -200.0),
        ("data_usage", -30.0),
        ("engagement", -0.5),
    ])
    def test_predict_negative_values(self, client, valid_customer_data, field, value):
        """Test that negative values are rejected."""
        invalid_data = valid_customer_data.copy()
        invalid_data[field] = value
        
        response = client.post("/predict/", json=invalid_data)
        
        assert response.status_code == 422


class TestPredictBatchEndpoint:
    """Test suite for batch prediction endpoint."""
    
    def test_predict_batch_valid(self, client, batch_customer_data):
        """Test batch prediction with valid data."""
        response = client.post("/predict/batch", json=batch_customer_data)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_predictions" in data
        assert "high_risk_count" in data
        assert "processing_time_seconds" in data
        
        assert data["total_predictions"] == len(batch_customer_data["customers"])
        assert len(data["predictions"]) == data["total_predictions"]
        
        # Verify each prediction has required fields
        for pred in data["predictions"]:
            assert "customer_id" in pred
            assert "churn_prediction" in pred
            assert "churn_probability" in pred
            assert "risk_level" in pred
    
    def test_predict_batch_empty(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"customers": []})
        
        assert response.status_code == 422  # Empty list not allowed
    
    def test_predict_batch_exceeds_limit(self, client, generate_random_customer):
        """Test batch prediction exceeds size limit."""
        # Generate 101 customers (limit is 100)
        large_batch = {
            "customers": [generate_random_customer() for _ in range(101)]
        }
        
        response = client.post("/predict/batch", json=large_batch)
        
        assert response.status_code == 400  # Bad request
        
        data = response.json()
        assert data["error"] == "BatchSizeExceededError"
        assert data["details"]["batch_size"] == 101
        assert data["details"]["limit"] == 100
    
    def test_predict_batch_partial_invalid(self, client, valid_customer_data, invalid_customer_data_age):
        """Test batch with some invalid customers."""
        mixed_batch = {
            "customers": [
                valid_customer_data,
                invalid_customer_data_age  # Invalid
            ]
        }
        
        response = client.post("/predict/batch", json=mixed_batch)
        
        # Should fail validation on the invalid customer
        assert response.status_code == 422
    
    def test_predict_batch_single_customer(self, client, valid_customer_data):
        """Test batch prediction with single customer."""
        single_batch = {"customers": [valid_customer_data]}
        
        response = client.post("/predict/batch", json=single_batch)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_predictions"] == 1


class TestModelInfoEndpoint:
    """Test suite for model info endpoint."""
    
    def test_model_info(self, client):
        """Test model info endpoint."""
        response = client.get("/predict/model-info")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "model_name" in data
        assert "model_version" in data
        assert "model_stage" in data
        assert "model_loaded" in data
        assert "feature_pipeline_loaded" in data
        assert "threshold" in data
        
        assert data["model_name"] == "churnguard-classifier"
        assert data["model_stage"] in ["None", "Staging", "Production", "Archived"]
        assert data["model_loaded"] is True
        assert data["feature_pipeline_loaded"] is True
        assert 0.0 <= data["threshold"] <= 1.0


class TestPredictionPerformance:
    """Performance tests for prediction endpoints."""
    
    @pytest.mark.slow
    def test_single_prediction_latency(self, client, valid_customer_data):
        """Test single prediction response time."""
        import time
        
        start = time.time()
        response = client.post("/predict/", json=valid_customer_data)
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 1.0  # Should complete in < 1 second
    
    @pytest.mark.slow
    def test_batch_prediction_throughput(self, client, generate_random_customer):
        """Test batch prediction throughput."""
        import time
        
        # Create batch of 50 customers
        batch = {
            "customers": [generate_random_customer() for _ in range(50)]
        }
        
        start = time.time()
        response = client.post("/predict/batch", json=batch)
        duration = time.time() - start
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_predictions"] == 50
        
        # Should process 50 customers in < 5 seconds
        assert duration < 5.0
        
        # Calculate throughput
        throughput = 50 / duration
        print(f"\nBatch throughput: {throughput:.2f} predictions/second")


class TestRootEndpoint:
    """Test suite for root endpoint."""
    
    def test_root_endpoint(self, client):
        """Test API root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "environment" in data
        assert "endpoints" in data
        
        # Check endpoints are documented
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "predict_single" in endpoints
        assert "predict_batch" in endpoints
        assert "model_info" in endpoints