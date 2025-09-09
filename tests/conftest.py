"""
Pytest configuration and shared fixtures.
"""

import os
import tempfile
from collections.abc import Generator

import pandas as pd
import pytest
from statsforecast import StatsForecast  # type: ignore
from statsforecast.models import _TS, AutoETS, Naive  # type: ignore
from statsforecast.utils import ConformalIntervals  # type: ignore

from src.fcd.model.time_series import save_model, simulate_series, train_model

D = dict[str, dict[str, float]]
BO = tuple[int, int, int]
BSO = tuple[int, int, int, int]


@pytest.fixture
def config() -> tuple[int, BO, BSO, D, list[_TS], int]:
    NOBS = 100  # Increased to have more data for testing
    BASE_ORDER = (1, 0, 1)
    BASE_SEASONAL_ORDER = (1, 0, 1, 7)
    OOT_SAMPLE_SIZE = 30
    intervals = ConformalIntervals(h=OOT_SAMPLE_SIZE, n_windows=2)
    MODELS = [
        Naive(prediction_intervals=intervals),
        AutoETS(prediction_intervals=intervals, season_length=7),
    ]
    DEMAND_LEVELS = {
        "very_low": {
            "count": 1,
            "base_seed": 42,
            "intercept": 25.0,
            "slope_base": 0.01,
            "phi_1": 0.8,
            "theta_1": 0.5,
            "Phi_1": 0.5,
            "Theta_1": 0.3,
            "variance": 50.0,
        },
    }

    return (
        NOBS,
        BASE_ORDER,
        BASE_SEASONAL_ORDER,
        DEMAND_LEVELS,
        MODELS,
        OOT_SAMPLE_SIZE,
    )


@pytest.fixture
def sample_models(
    config: tuple[int, BO, BSO, D, list[_TS], int],
) -> dict[str, _TS]:
    """Create trained models for testing."""
    (
        NOBS,
        BASE_ORDER,
        BASE_SEASONAL_ORDER,
        DEMAND_LEVELS,
        MODELS,
        OOT_SAMPLE_SIZE,
    ) = config

    return train_model(
        nobs=NOBS,
        base_order=BASE_ORDER,
        base_seasonal_order=BASE_SEASONAL_ORDER,
        demand_levels=DEMAND_LEVELS,
        models=MODELS,
        out_of_sample_size=OOT_SAMPLE_SIZE,
    )


@pytest.fixture
def sample_series_df(config: tuple[int, BO, BSO, D]) -> pd.DataFrame:
    """Sample simulated series data."""
    NOBS, BASE_ORDER, BASE_SEASONAL_ORDER, DEMAND_LEVELS, _, _ = config

    return simulate_series(
        n_obs=NOBS,
        base_order=BASE_ORDER,
        base_seasonal_order=BASE_SEASONAL_ORDER,
        demand_levels=DEMAND_LEVELS,
    )


@pytest.fixture
def temp_models_file(
    sample_models: dict[str, _TS],
) -> Generator[str, None, None]:
    """Create a temporary models file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        save_model(sample_models, tmp_file.name)

        yield tmp_file.name

    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def sample_prediction_request() -> dict[str, any]:
    """Sample prediction request data."""
    return {"h": 7, "level": [95], "unique_id": "0"}


@pytest.fixture
def sample_statsforecast(
    sample_series_df: pd.DataFrame,
    config: tuple[int, BO, BSO, D, list[_TS], int],
) -> StatsForecast:
    """Create a sample StatsForecast object for testing."""
    _, _, _, _, _, OOT_SAMPLE_SIZE = config
    train_df = sample_series_df.iloc[:-OOT_SAMPLE_SIZE, :]
    intervals = ConformalIntervals(h=OOT_SAMPLE_SIZE, n_windows=2)
    statsforecast = StatsForecast(
        models=[
            Naive(prediction_intervals=intervals),
            AutoETS(prediction_intervals=intervals, season_length=7),
        ],
        freq="D",
        n_jobs=1,
    )
    print(train_df)
    statsforecast.fit(train_df)

    return statsforecast


@pytest.fixture
def sample_test_df(sample_series_df: pd.DataFrame) -> pd.DataFrame:
    """Create sample test data for championship testing."""
    return sample_series_df.iloc[61:, :]
