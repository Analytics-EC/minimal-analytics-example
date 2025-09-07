"""
Integration tests for FastAPI application.
"""
import concurrent.futures
import time
from collections.abc import AsyncGenerator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from src.fcd.api.main import app, lifespan


@pytest.fixture
async def test_app() -> AsyncGenerator[FastAPI, None]:
    async with lifespan(app):
        yield app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.fixture
def sample_prediction_request() -> dict[str, list[float]]:
    """Sample prediction request data."""
    return {"features": [6.0]}


@pytest.fixture
def invalid_prediction_request() -> dict[str, str]:
    """Invalid prediction request data."""
    return {"features": "invalid"}


class TestHomeEndpoint:
    """Test the home endpoint."""
    @pytest.mark.integration
    def test_home_endpoint(self, client: TestClient) -> None:
        """Test that home endpoint returns correct response."""
        response = client.get("/")
        data = response.json()

        assert response.status_code == 200
        assert "message" in data
        assert data["message"] == "API for Linear Regression Model"


class TestHealthEndpoint:
    """Test the health check endpoint."""

    @pytest.mark.integration
    def test_health_endpoint(self, client: TestClient) -> None:
        """Test that health endpoint returns correct response."""
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)

    @pytest.mark.integration
    def test_health_model_loaded(self, client: TestClient) -> None:
        """Test that health endpoint indicates model is loaded."""
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert data["model_loaded"] is True


class TestPredictEndpoint:
    """Test the prediction endpoint."""

    @pytest.mark.integration
    def test_predict_success(
            self,
            client: TestClient,
            sample_prediction_request: dict[str, list[float]]) -> None:
        """Test successful prediction request."""
        response = client.post("/predict", json=sample_prediction_request)
        data = response.json()

        assert response.status_code == 200
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] > 0

    @pytest.mark.integration
    def test_predict_multiple_features(self, client: TestClient) -> None:
        """Test prediction with multiple features should return error."""
        request_data = {"features": [6.0, 7.0]}
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Should fail since model expects 1 feature
        assert response.status_code == 400
        assert "detail" in data

    @pytest.mark.integration
    def test_predict_empty_features(self, client: TestClient) -> None:
        """Test prediction with empty features list."""
        request_data: dict[str, list[float]] = {"features": []}
        response = client.post("/predict", json=request_data)
        data = response.json()

        assert response.status_code == 400
        assert "detail" in data

    @pytest.mark.integration
    def test_predict_invalid_json(self, client: TestClient) -> None:
        """Test prediction with invalid JSON."""
        response = client.post("/predict", json={"invalid": "data"})

        # Validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_missing_features(self, client: TestClient) -> None:
        """Test prediction without features field."""
        response = client.post("/predict", json={})

        # Validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_predict_wrong_method(self, client: TestClient) -> None:
        """Test prediction endpoint with wrong HTTP method."""
        response = client.get("/predict")

        # Method not allowed
        assert response.status_code == 405

    @pytest.mark.integration
    def test_predict_consistency(
            self,
            client: TestClient,
            sample_prediction_request: dict[str, list[float]]) -> None:
        """Test that multiple identical requests return same prediction."""

        response1 = client.post("/predict", json=sample_prediction_request)
        response2 = client.post("/predict", json=sample_prediction_request)
        data1 = response1.json()
        data2 = response2.json()

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert data1["prediction"] == data2["prediction"]


class TestAPIErrorHandling:
    """Test API error handling."""

    @pytest.mark.integration
    def test_nonexistent_endpoint(self, client: TestClient) -> None:
        """Test accessing nonexistent endpoint."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_predict_with_non_numeric_features(self, client: TestClient) -> None:
        """Test prediction with non-numeric features."""
        request_data = {"features": ["invalid", "data"]}
        response = client.post("/predict", json=request_data)
        data = response.json()

        # Validation error for non-numeric data
        assert response.status_code == 422
        assert "detail" in data


class TestAPIPerformance:
    """Test API performance characteristics."""

    @pytest.mark.integration
    def test_multiple_concurrent_requests(
            self,
            client: TestClient,
            sample_prediction_request: dict[str, list[float]]) -> None:
        """Test handling multiple concurrent requests."""

        def make_request() -> Response:
            return client.post("/predict", json=sample_prediction_request)

        start_time = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]

        end_time = time.time()

        # All requests should succeed
        for response in responses:
            data = response.json()

            assert response.status_code == 200
            assert "prediction" in data

        # Should complete in reasonable time (less than 5 seconds)
        assert (end_time - start_time) < 5.0

    @pytest.mark.integration
    def test_response_time(
            self,
            client: TestClient,
            sample_prediction_request: dict[str, list[float]]) -> None:
        """Test that response time is reasonable."""

        start_time = time.time()
        response = client.post("/predict", json=sample_prediction_request)
        end_time = time.time()
        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 1.0


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    @pytest.mark.integration
    def test_openapi_docs(self, client: TestClient) -> None:
        """Test that OpenAPI documentation is accessible."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.integration
    def test_openapi_json(self, client: TestClient) -> None:
        """Test that OpenAPI JSON schema is accessible."""
        response = client.get("/openapi.json")
        data = response.json()

        assert response.status_code == 200
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data


class TestAPIVersioning:
    """Test API versioning and metadata."""

    @pytest.mark.integration
    def test_api_info(self, client: TestClient) -> None:
        """Test that API info is correctly set."""
        response = client.get("/openapi.json")
        data = response.json()

        assert response.status_code == 200
        assert data["info"]["title"] == "Regression API"
        assert data["info"]["version"] == "0.1.0"
        assert "description" in data["info"]
