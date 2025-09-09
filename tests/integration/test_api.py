"""
Integration tests for FastAPI application.
"""

import concurrent.futures
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from src.fcd.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_prediction_request() -> dict[str, Any]:
    """Sample prediction request data."""
    return {"h": 7, "level": [95], "unique_id": "0"}


@pytest.fixture
def invalid_prediction_request() -> dict[str, Any]:
    """Invalid prediction request data."""
    return {"h": "invalid", "level": [95], "unique_id": "0"}


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
        assert "models_loaded" in data
        assert data["status"] == "healthy"
        assert isinstance(data["models_loaded"], bool)

    @pytest.mark.integration
    def test_health_models_loaded(self, client: TestClient) -> None:
        """Test that health endpoint indicates models are loaded."""
        response = client.get("/health")
        data = response.json()

        assert response.status_code == 200
        assert data["models_loaded"] is True


class TestTimeSeriesPredictEndpoint:
    """Test the time series prediction endpoint."""

    @pytest.mark.integration
    def test_ts_predict_success(
        self, client: TestClient, sample_prediction_request: dict[str, Any]
    ) -> None:
        """Test successful time series prediction request."""
        response = client.post("/ts/predict", json=sample_prediction_request)
        data = response.json()

        assert response.status_code == 200
        assert "prediction" in data
        assert isinstance(data["prediction"], dict)
        assert len(data["prediction"]) > 0

    @pytest.mark.integration
    def test_ts_predict_invalid_h(self, client: TestClient) -> None:
        """Test prediction with invalid horizon."""
        request_data = {"h": -1, "level": [95], "unique_id": "0"}
        response = client.post("/ts/predict", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    @pytest.mark.integration
    def test_ts_predict_invalid_level(self, client: TestClient) -> None:
        """Test prediction with invalid level."""
        request_data = {
            "h": 7,
            "level": [200],  # Invalid level > 100
            "unique_id": "0",
        }
        response = client.post("/ts/predict", json=request_data)

        # Should fail validation
        assert response.status_code == 422

    @pytest.mark.integration
    def test_ts_predict_invalid_unique_id(self, client: TestClient) -> None:
        """Test prediction with invalid unique_id."""
        request_data = {
            "h": 7,
            "level": [95],
            "unique_id": "999",  # Non-existent unique_id
        }
        response = client.post("/ts/predict", json=request_data)
        data = response.json()

        assert response.status_code == 422
        assert "detail" in data
        assert "Model not found" in data["detail"]

    @pytest.mark.integration
    def test_ts_predict_missing_fields(self, client: TestClient) -> None:
        """Test prediction with missing required fields."""
        # Missing h
        response = client.post(
            "/ts/predict", json={"level": [95], "unique_id": "0"}
        )
        assert response.status_code == 422

        # Missing level
        response = client.post("/ts/predict", json={"h": 7, "unique_id": "0"})
        assert response.status_code == 422

        # Missing unique_id
        response = client.post("/ts/predict", json={"h": 7, "level": [95]})
        assert response.status_code == 422

    @pytest.mark.integration
    def test_ts_predict_invalid_json(self, client: TestClient) -> None:
        """Test prediction with invalid JSON."""
        response = client.post("/ts/predict", json={"invalid": "data"})

        # Validation error
        assert response.status_code == 422

    @pytest.mark.integration
    def test_ts_predict_wrong_method(self, client: TestClient) -> None:
        """Test prediction endpoint with wrong HTTP method."""
        response = client.get("/ts/predict")

        # Method not allowed
        assert response.status_code == 405

    @pytest.mark.integration
    def test_ts_predict_consistency(
        self, client: TestClient, sample_prediction_request: dict[str, Any]
    ) -> None:
        """Test that multiple identical requests return same prediction."""
        response1 = client.post("/ts/predict", json=sample_prediction_request)
        response2 = client.post("/ts/predict", json=sample_prediction_request)
        data1 = response1.json()
        data2 = response2.json()

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert data1["prediction"] == data2["prediction"]

    @pytest.mark.integration
    def test_ts_predict_different_horizons(self, client: TestClient) -> None:
        """Test predictions with different horizons."""
        request_7 = {"h": 7, "level": [95], "unique_id": "0"}
        request_14 = {"h": 14, "level": [95], "unique_id": "0"}

        response_7 = client.post("/ts/predict", json=request_7)
        response_14 = client.post("/ts/predict", json=request_14)

        assert response_7.status_code == 200
        assert response_14.status_code == 200

        data_7 = response_7.json()
        data_14 = response_14.json()

        # Different horizons should produce different results
        assert data_7["prediction"] != data_14["prediction"]

    @pytest.mark.integration
    def test_ts_predict_different_series(self, client: TestClient) -> None:
        """Test predictions for different time series."""
        request_0 = {"h": 7, "level": [95], "unique_id": "0"}
        request_1 = {"h": 7, "level": [95], "unique_id": "1"}

        response_0 = client.post("/ts/predict", json=request_0)
        response_1 = client.post("/ts/predict", json=request_1)

        assert response_0.status_code == 200
        assert response_1.status_code == 200

        data_0 = response_0.json()
        data_1 = response_1.json()

        # Different series should produce different results
        assert data_0["prediction"] != data_1["prediction"]


class TestAPIErrorHandling:
    """Test API error handling."""

    @pytest.mark.integration
    def test_nonexistent_endpoint(self, client: TestClient) -> None:
        """Test accessing nonexistent endpoint."""
        response = client.get("/nonexistent")

        assert response.status_code == 404

    @pytest.mark.integration
    def test_ts_predict_with_non_numeric_h(self, client: TestClient) -> None:
        """Test prediction with non-numeric horizon."""
        request_data = {"h": "invalid", "level": [95], "unique_id": "0"}
        response = client.post("/ts/predict", json=request_data)

        # Validation error for non-numeric data
        assert response.status_code == 422

    @pytest.mark.integration
    def test_ts_predict_with_non_list_level(self, client: TestClient) -> None:
        """Test prediction with non-list level."""
        request_data = {
            "h": 7,
            "level": 95,  # Should be a list
            "unique_id": "0",
        }
        response = client.post("/ts/predict", json=request_data)

        # Validation error
        assert response.status_code == 422


class TestAPIPerformance:
    """Test API performance characteristics."""

    @pytest.mark.integration
    def test_multiple_concurrent_requests(
        self, client: TestClient, sample_prediction_request: dict[str, Any]
    ) -> None:
        """Test handling multiple concurrent requests."""

        def make_request() -> Response:
            return client.post("/ts/predict", json=sample_prediction_request)

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

        # Should complete in reasonable time (less than 10 seconds for time series)
        assert (end_time - start_time) < 10.0

    @pytest.mark.integration
    def test_response_time(
        self, client: TestClient, sample_prediction_request: dict[str, Any]
    ) -> None:
        """Test that response time is reasonable."""

        start_time = time.time()
        response = client.post("/ts/predict", json=sample_prediction_request)
        end_time = time.time()
        response_time = end_time - start_time

        assert response.status_code == 200
        assert response_time < 5.0  # Time series predictions might take longer


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

    @pytest.mark.integration
    def test_redoc_docs(self, client: TestClient) -> None:
        """Test that ReDoc documentation is accessible."""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


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

    @pytest.mark.integration
    def test_api_tags(self, client: TestClient) -> None:
        """Test that API tags are correctly set."""
        response = client.get("/openapi.json")
        data = response.json()

        assert response.status_code == 200

        # Check that time_series tag exists
        paths = data["paths"]
        ts_paths = [path for path in paths.keys() if path.startswith("/ts")]
        assert len(ts_paths) > 0

        # Check that time_series tag is used
        for path in ts_paths:
            for method in paths[path].keys():
                if method != "parameters":
                    tags = paths[path][method].get("tags", [])
                    assert "time_series" in tags
