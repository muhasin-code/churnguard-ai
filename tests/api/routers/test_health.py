"""
Tests for health check endpoints.

Tests:
- GET /health - Basic health check
- GET /health/ready - Readiness probe
- GET /health/live - Liveness probe
"""

import pytest


class TestHealthEndpoints:
    """Test suite for health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0.0"
        assert data["environment"] in ["development", "production"]
        assert data["model_loaded"] is True
        assert "dependencies" in data
        
        # Check dependencies
        deps = data["dependencies"]
        assert "python" in deps
        assert "fastapi" in deps
        assert "mlflow" in deps
    
    def test_readiness_check(self, client):
        """Test readiness probe endpoint."""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "ready" in data
        assert "checks" in data
        assert "message" in data
        
        # Check individual readiness checks
        checks = data["checks"]
        assert checks["config_loaded"] is True
        assert checks["model_loaded"] is True
    
    def test_liveness_check(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["alive"] is True
    
    def test_health_without_trailing_slash(self, client):
        """Test health endpoint redirects without trailing slash."""
        response = client.get("/health", follow_redirects=False)
        
        # Should redirect to /health/
        assert response.status_code == 307
    
    def test_health_response_time(self, client):
        """Test health check responds quickly."""
        import time
        
        start = time.time()
        response = client.get("/health/")
        duration = time.time() - start
        
        assert response.status_code == 200
        assert duration < 0.5  # Should respond in < 500ms