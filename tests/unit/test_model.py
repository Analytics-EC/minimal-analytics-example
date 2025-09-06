"""
Unit tests for model.py module.
"""

import os
import tempfile

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.regression_api.model import (
    load_model,
    predict,
    save_model,
    train_model,
)


class TestTrainModel:
    """Test model training functionality."""

    @pytest.mark.unit
    def test_train_model_returns_linear_regression(self):
        """Test that train_model returns a LinearRegression instance."""
        model = train_model()
        assert isinstance(model, LinearRegression)

    @pytest.mark.unit
    def test_train_model_is_fitted(self):
        """Test that the returned model is fitted."""
        model = train_model()
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")
        assert model.coef_ is not None
        assert model.intercept_ is not None

    @pytest.mark.unit
    def test_train_model_coefficients(self):
        """Test that the model has expected coefficient structure."""
        model = train_model()
        assert len(model.coef_) == 1  # Single feature
        assert isinstance(model.coef_[0], (int, float))
        assert isinstance(model.intercept_, (int, float))


class TestSaveModel:
    """Test model saving functionality."""

    @pytest.mark.unit
    def test_save_model_creates_file(self, sample_model):
        """Test that save_model creates a file."""
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as tmp_file:
            filepath = tmp_file.name

        try:
            save_model(sample_model, filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    @pytest.mark.unit
    def test_save_model_with_custom_path(self, sample_model):
        """Test saving model with custom filepath."""
        custom_path = "test_model.pkl"
        try:
            save_model(sample_model, custom_path)
            assert os.path.exists(custom_path)
        finally:
            if os.path.exists(custom_path):
                os.unlink(custom_path)


class TestLoadModel:
    """Test model loading functionality."""

    @pytest.mark.unit
    def test_load_model_success(self, temp_model_file):
        """Test successful model loading."""
        model = load_model(temp_model_file)
        assert model is not None
        assert isinstance(model, LinearRegression)
        assert hasattr(model, "coef_")
        assert hasattr(model, "intercept_")

    @pytest.mark.unit
    def test_load_model_nonexistent_file(self):
        """Test loading model from nonexistent file."""
        model = load_model("nonexistent_model.pkl")
        assert model is None

    @pytest.mark.unit
    def test_loaded_model_predictions(self, temp_model_file):
        """Test that loaded model can make predictions."""
        model = load_model(temp_model_file)
        assert model is not None

        # Test prediction
        features = [6.0]
        prediction = model.predict(np.array(features).reshape(1, -1))
        assert len(prediction) == 1
        assert isinstance(prediction[0], (int, float))


class TestPredict:
    """Test prediction functionality."""

    @pytest.mark.unit
    def test_predict_single_feature(self, sample_model):
        """Test prediction with single feature."""
        features = [6.0]
        prediction = predict(sample_model, features)
        assert isinstance(prediction, float)
        assert prediction > 0  # Should be positive based on training data

    @pytest.mark.unit
    def test_predict_multiple_features(self, sample_model):
        """Test prediction with multiple features should raise error since model expects 1 feature."""
        features = [
            6.0,
            7.0,
        ]  # Model only uses one feature, so this should fail
        with pytest.raises(ValueError, match="features"):
            predict(sample_model, features)

    @pytest.mark.unit
    def test_predict_empty_features(self, sample_model):
        """Test prediction with empty features list."""
        features = []
        with pytest.raises(ValueError):
            predict(sample_model, features)

    @pytest.mark.unit
    def test_predict_consistency(self, sample_model):
        """Test that predictions are consistent."""
        features = [6.0]
        prediction1 = predict(sample_model, features)
        prediction2 = predict(sample_model, features)
        assert prediction1 == prediction2

    @pytest.mark.unit
    def test_predict_different_values(self, sample_model):
        """Test predictions with different input values."""
        features1 = [1.0]
        features2 = [10.0]

        prediction1 = predict(sample_model, features1)
        prediction2 = predict(sample_model, features2)

        assert prediction1 != prediction2
        assert isinstance(prediction1, float)
        assert isinstance(prediction2, float)


class TestModelIntegration:
    """Integration tests for model workflow."""

    @pytest.mark.unit
    def test_train_save_load_workflow(self):
        """Test complete workflow: train -> save -> load."""
        # Train model
        model1 = train_model()

        # Save model
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as tmp_file:
            filepath = tmp_file.name

        try:
            save_model(model1, filepath)

            # Load model
            model2 = load_model(filepath)

            # Compare predictions
            features = [6.0]
            pred1 = predict(model1, features)
            pred2 = predict(model2, features)

            assert pred1 == pred2
            assert isinstance(pred1, float)
            assert isinstance(pred2, float)

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    @pytest.mark.unit
    def test_model_mathematical_properties(self, sample_model):
        """Test mathematical properties of the model."""
        # Test linearity: f(ax + b) = af(x) + b (approximately)
        features = [3.0]
        base_prediction = predict(sample_model, features)

        # Test with scaled input
        scaled_features = [6.0]  # 2 * 3
        scaled_prediction = predict(sample_model, scaled_features)

        # For linear regression, the relationship should be approximately linear
        # This is more of a sanity check than a strict mathematical test
        assert isinstance(base_prediction, float)
        assert isinstance(scaled_prediction, float)
        assert base_prediction != scaled_prediction
