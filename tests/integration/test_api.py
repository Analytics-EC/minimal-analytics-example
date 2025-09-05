"""
Integration tests for FastAPI application.
"""
import pytest
import httpx
import os
import tempfile
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.regression_api.model import train_model, save_model, load_model, predict
from pydantic import BaseModel
from typing import List


@pytest.fixture
def test_app():
    """Create a test FastAPI app with pre-loaded model."""
    # Create and train a model
    model = train_model()
    
    # Create test app
    app = FastAPI(
        title="Regression API Test",
        description="Test API for linear regression predictions",
        version="0.1.0"
    )
    
    # Pydantic models
    class PredictionRequest(BaseModel):
        features: List[float]

    class PredictionResponse(BaseModel):
        prediction: float

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
    
    @app.get("/", response_model=dict)
    async def home():
        return {"message": "API for Linear Regression Model"}
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            model_loaded=model is not None
        )
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_endpoint(request: PredictionRequest):
        if model is None:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        try:
            prediction = predict(model, request.features)
            return PredictionResponse(prediction=prediction)
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=400, detail=str(e))
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request data."""
    return {"features": [6.0]}


@pytest.fixture
def invalid_prediction_request():
    """Invalid prediction request data."""
    return {"features": "invalid"}


class TestHomeEndpoint:
    """Test the home endpoint."""
    
    @pytest.mark.integration
    def test_home_endpoint(self, client):
        """Test that home endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["message"] == "API for Linear Regression Model"


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    @pytest.mark.integration
    def test_health_endpoint(self, client):
        """Test that health endpoint returns correct response."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
    
    @pytest.mark.integration
    def test_health_model_loaded(self, client):
        """Test that health endpoint indicates model is loaded."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Test the prediction endpoint."""
    
    @pytest.mark.integration
    def test_predict_success(self, client, sample_prediction_request):
        """Test successful prediction request."""
        response = client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] > 0
    
    @pytest.mark.integration
    def test_predict_multiple_features(self, client):
        """Test prediction with multiple features should return error."""
        request_data = {"features": [6.0, 7.0]}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400  # Should fail since model expects 1 feature
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.integration
    def test_predict_empty_features(self, client):
        """Test prediction with empty features list."""
        request_data = {"features": []}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.integration
    def test_predict_invalid_json(self, client):
        """Test prediction with invalid JSON."""
        response = client.post("/predict", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.integration
    def test_predict_missing_features(self, client):
        """Test prediction without features field."""
        response = client.post("/predict", json={})
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.integration
    def test_predict_wrong_method(self, client, sample_prediction_request):
        """Test prediction endpoint with wrong HTTP method."""
        response = client.get("/predict")
        assert response.status_code == 405  # Method not allowed
    
    @pytest.mark.integration
    def test_predict_consistency(self, client, sample_prediction_request):
        """Test that multiple identical requests return same prediction."""
        response1 = client.post("/predict", json=sample_prediction_request)
        response2 = client.post("/predict", json=sample_prediction_request)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["prediction"] == data2["prediction"]


class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.mark.integration
    def test_nonexistent_endpoint(self, client):
        """Test accessing nonexistent endpoint."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    @pytest.mark.integration
    def test_predict_with_non_numeric_features(self, client):
        """Test prediction with non-numeric features."""
        request_data = {"features": ["invalid", "data"]}
        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error for non-numeric data
        data = response.json()
        assert "detail" in data


class TestAPIPerformance:
    """Test API performance characteristics."""
    
    @pytest.mark.integration
    def test_multiple_concurrent_requests(self, client, sample_prediction_request):
        """Test handling multiple concurrent requests."""
        import concurrent.futures
        import time
        
        def make_request():
            return client.post("/predict", json=sample_prediction_request)
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        end_time = time.time()
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
        
        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0
    
    @pytest.mark.integration
    def test_response_time(self, client, sample_prediction_request):
        """Test that response time is reasonable."""
        import time
        
        start_time = time.time()
        response = client.post("/predict", json=sample_prediction_request)
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.mark.integration
    def test_openapi_docs(self, client):
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    @pytest.mark.integration
    def test_openapi_json(self, client):
        """Test that OpenAPI JSON schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


class TestAPIVersioning:
    """Test API versioning and metadata."""
    
    @pytest.mark.integration
    def test_api_info(self, client):
        """Test that API info is correctly set."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        
        assert data["info"]["title"] == "Regression API Test"
        assert data["info"]["version"] == "0.1.0"
        assert "description" in data["info"]
