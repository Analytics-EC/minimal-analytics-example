"""
Pytest configuration and shared fixtures.
"""

import os
import tempfile
from collections.abc import Generator

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.fcd.model.linear_regression import save_model, train_model


@pytest.fixture
def sample_model() -> LinearRegression:
    """Create a trained model for testing."""
    return train_model()


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Sample training data."""
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([20, 35, 41, 50, 78])

    return X, y


@pytest.fixture
def temp_model_file(sample_model: LinearRegression) -> Generator[str, None, None]:
    """Create a temporary model file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        save_model(sample_model, tmp_file.name)

        yield tmp_file.name

    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def test_features() -> list[float]:
    """Sample features for prediction testing."""
    return [6.0]


@pytest.fixture
def expected_prediction() -> float:
    """Expected prediction for test features [6.0]."""
    # Based on the trained model, feature 6 should predict around 84.1
    return 84.1
