"""
Unit tests for time series model module.
"""

import os
import tempfile

import pandas as pd
import pytest
from statsforecast.models import _TS  # type: ignore

from src.fcd.model.config import (
    BASE_ORDER,
    BASE_SEASONAL_ORDER,
    DEMAND_LEVELS,
    NOBS,
)
from src.fcd.model.time_series import (
    championship,
    load_model,
    predict,
    save_model,
    simulate_series,
)


class TestSimulateSeries:
    """Test series simulation functionality."""

    @pytest.mark.unit
    def test_simulate_series_returns_dataframe(self) -> None:
        """Test that simulate_series returns a pandas DataFrame."""
        series_df = simulate_series(
            n_obs=NOBS,
            base_order=BASE_ORDER,
            base_seasonal_order=BASE_SEASONAL_ORDER,
            demand_levels=DEMAND_LEVELS,
        )

        assert isinstance(series_df, pd.DataFrame)
        assert len(series_df) > 0

    @pytest.mark.unit
    def test_simulate_series_has_required_columns(self) -> None:
        """Test that simulated series has required columns."""
        series_df = simulate_series(
            n_obs=NOBS,
            base_order=BASE_ORDER,
            base_seasonal_order=BASE_SEASONAL_ORDER,
            demand_levels=DEMAND_LEVELS,
        )

        required_columns = ["unique_id", "ds", "y"]
        for col in required_columns:
            assert col in series_df.columns

    @pytest.mark.unit
    def test_simulate_series_unique_ids(self) -> None:
        """Test that unique_ids are properly generated."""
        series_df = simulate_series(
            n_obs=NOBS,
            base_order=BASE_ORDER,
            base_seasonal_order=BASE_SEASONAL_ORDER,
            demand_levels=DEMAND_LEVELS,
        )

        # Should have 10 unique series (2+3+3+2)
        unique_ids = series_df["unique_id"].unique()
        assert len(unique_ids) == 10

        # All unique_ids should be strings
        for unique_id in unique_ids:
            assert isinstance(unique_id, str)

    @pytest.mark.unit
    def test_simulate_series_date_range(self) -> None:
        """Test that date range is properly generated."""
        series_df = simulate_series(
            n_obs=NOBS,
            base_order=BASE_ORDER,
            base_seasonal_order=BASE_SEASONAL_ORDER,
            demand_levels=DEMAND_LEVELS,
        )

        # Check date range
        assert series_df["ds"].min() == pd.Timestamp("2022-01-01")
        assert len(series_df["ds"].unique()) == NOBS

        # Check that each series has the same number of observations
        for unique_id in series_df["unique_id"].unique():
            series_data = series_df[series_df["unique_id"] == unique_id]
            assert len(series_data) == NOBS


class TestTrainModel:
    """Test model training functionality."""

    @pytest.mark.unit
    def test_train_model_returns_dict(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test that train_model returns a dictionary of models."""
        models = sample_models

        print(models)

        assert isinstance(models, dict)
        assert len(models) > 0

    @pytest.mark.unit
    def test_train_model_has_expected_keys(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test that trained models have expected unique_ids as keys."""
        models = sample_models

        # Should have 10 models (one for each series)
        assert len(models) == 1

        # Keys should be strings representing unique_ids
        for key in models.keys():
            assert isinstance(key, str)
            assert key.isdigit()  # unique_ids are numeric strings

    @pytest.mark.unit
    def test_train_model_values_are_ts_models(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test that trained model values are _TS model instances."""
        models = sample_models

        for model in models.values():
            assert hasattr(model, "predict")  # _TS models have predict method


class TestSaveModel:
    """Test model saving functionality."""

    @pytest.mark.unit
    def test_save_model_creates_file(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test that save_model creates a file."""
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as tmp_file:
            filepath = tmp_file.name

        try:
            save_model(sample_models, filepath)
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    @pytest.mark.unit
    def test_save_model_with_custom_path(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test saving model with custom filepath."""
        custom_path = "test_models.pkl"
        try:
            save_model(sample_models, custom_path)
            assert os.path.exists(custom_path)
        finally:
            if os.path.exists(custom_path):
                os.unlink(custom_path)


class TestLoadModel:
    """Test model loading functionality."""

    @pytest.mark.unit
    def test_load_model_success(self, temp_models_file: str) -> None:
        """Test successful model loading."""
        models = load_model(temp_models_file)

        assert models is not None
        assert isinstance(models, dict)
        assert len(models) > 0

    @pytest.mark.unit
    def test_load_model_nonexistent_file(self) -> None:
        """Test loading model from nonexistent file."""
        models = load_model("nonexistent_models.pkl")

        assert models is None

    @pytest.mark.unit
    def test_loaded_models_structure(self, temp_models_file: str) -> None:
        """Test that loaded models have correct structure."""
        models = load_model(temp_models_file)

        assert models is not None

        for unique_id, model in models.items():
            assert isinstance(unique_id, str)
            assert hasattr(model, "predict")


class TestPredict:
    """Test prediction functionality."""

    @pytest.mark.unit
    def test_predict_success(self, sample_models: dict[str, _TS]) -> None:
        """Test successful prediction."""
        unique_id = list(sample_models.keys())[0]
        h = 7
        level = [95]

        prediction = predict(sample_models, h, level, unique_id)

        assert isinstance(prediction, dict)
        assert len(prediction) > 0

    @pytest.mark.unit
    def test_predict_invalid_unique_id(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test prediction with invalid unique_id."""
        h = 7
        level = [95]
        invalid_unique_id = "999"

        with pytest.raises(ValueError, match="Model not found"):
            predict(sample_models, h, level, invalid_unique_id)

    @pytest.mark.unit
    def test_predict_different_horizons(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test predictions with different horizons."""
        unique_id = list(sample_models.keys())[0]
        level = [95]
        print(sample_models)

        prediction_7 = predict(sample_models, 7, level, unique_id)
        prediction_14 = predict(sample_models, 14, level, unique_id)

        # Different horizons should produce different results
        assert prediction_7 != prediction_14

    @pytest.mark.unit
    def test_predict_consistency(self, sample_models: dict[str, _TS]) -> None:
        """Test that predictions are consistent for same inputs."""
        unique_id = list(sample_models.keys())[0]
        h = 7
        level = [95]

        prediction1 = predict(sample_models, h, level, unique_id)
        prediction2 = predict(sample_models, h, level, unique_id)

        # Predictions should be identical for same inputs
        assert prediction1 == prediction2


class TestChampionship:
    """Test championship functionality."""

    @pytest.mark.unit
    def test_championship_returns_dataframe(
        self, sample_statsforecast, sample_test_df
    ) -> None:
        """Test that championship returns a DataFrame."""
        print(sample_statsforecast, sample_test_df)
        result = championship(sample_statsforecast, sample_test_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    @pytest.mark.unit
    def test_championship_has_required_columns(
        self, sample_statsforecast, sample_test_df
    ) -> None:
        """Test that championship result has required columns."""
        result = championship(sample_statsforecast, sample_test_df)

        required_columns = ["unique_id", "model"]
        for col in required_columns:
            assert col in result.columns

    @pytest.mark.unit
    def test_championship_ranks_models(
        self, sample_statsforecast, sample_test_df
    ) -> None:
        """Test that championship properly ranks models."""
        result = championship(sample_statsforecast, sample_test_df)

        # Should have one champion per unique_id
        assert len(result) == len(sample_test_df["unique_id"].unique())


class TestModelIntegration:
    """Integration tests for model workflow."""

    @pytest.mark.unit
    def test_train_save_load_workflow(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test complete workflow: train -> save -> load."""
        # Train models
        models1 = sample_models

        # Save models
        with tempfile.NamedTemporaryFile(
            suffix=".pkl", delete=False
        ) as tmp_file:
            filepath = tmp_file.name

        try:
            save_model(models1, filepath)

            # Load models
            models2 = load_model(filepath)

            # Compare structure
            assert models2 is not None
            assert len(models1) == len(models2)
            assert set(models1.keys()) == set(models2.keys())

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)

    @pytest.mark.unit
    def test_end_to_end_prediction_workflow(
        self, sample_models: dict[str, _TS]
    ) -> None:
        """Test end-to-end prediction workflow."""
        # Train models
        models = sample_models

        # Make predictions for all series
        unique_ids = list(models.keys())
        h = 7
        level = [95]

        predictions = {}
        for unique_id in unique_ids:
            prediction = predict(models, h, level, unique_id)
            predictions[unique_id] = prediction

        # All predictions should be successful
        assert len(predictions) == len(unique_ids)

        for _, prediction in predictions.items():
            assert isinstance(prediction, dict)
            assert len(prediction) > 0
