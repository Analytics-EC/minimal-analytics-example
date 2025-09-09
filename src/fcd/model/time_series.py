"""
Model training and loading utilities for time series.
"""

import os
import pickle
from typing import cast

import numpy as np
import pandas as pd
from statsforecast import StatsForecast  # type: ignore
from statsforecast.models import _TS, Naive  # type: ignore

from .config import (
    BASE_ORDER,
    BASE_SEASONAL_ORDER,
    DEMAND_LEVELS,
    NOBS,
    MODELS,
    OOT_SAMPLE_SIZE,
)
from .simulate_arima import simulate_sarima


def train_model(
        nobs: int = NOBS,
        base_order: tuple[int, int, int] = BASE_ORDER,
        base_seasonal_order: tuple[int, int, int, int] = BASE_SEASONAL_ORDER,
        demand_levels: dict[str, dict[str, float]] = DEMAND_LEVELS,
        models: list[_TS] = MODELS,
        out_of_sample_size: int = OOT_SAMPLE_SIZE) -> dict[str, _TS]:
    """
    Train the model.

    Args:
        nobs: The number of observations.
        base_order: The base order.
        base_seasonal_order: The base seasonal order.
        demand_levels: The demand levels.
    """
    series_df = simulate_series(
        n_obs=nobs,
        base_order=base_order,
        base_seasonal_order=base_seasonal_order,
        demand_levels=demand_levels,
    )

    train_df = series_df.iloc[:-out_of_sample_size, :]
    test_df = series_df.iloc[-out_of_sample_size:, ]

    statsforecast = StatsForecast(
        models=models,
        freq="D",
        fallback_model=Naive(),
        n_jobs=-1,
    )

    statsforecast.fit(train_df)

    champions = championship(statsforecast, test_df)

    print("Time series champions: ", champions)

    return {
        unique_id: model
        for models, champ, unique_id
        in zip(
            statsforecast.fitted_,
            champions.model,
            champions.unique_id,
            strict=True,
        )
        for model in models
        if model.alias == champ
    }


def simulate_series(
    n_obs: int,
    base_order: tuple[int, int, int],
    base_seasonal_order: tuple[int, int, int, int],
    demand_levels: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Simulate the series.

    Args:
        n_obs: The number of observations.
        base_order: The base order.
        base_seasonal_order: The base seasonal order.
        demand_levels: The demand levels.

    Returns:
        The simulated series.
    """
    # Generar todas las series con un solo loop
    all_series = []
    for _, level_config in demand_levels.items():
        for i in range(int(level_config["count"])):
            # Construir parámetros usando la configuración del nivel
            params = np.array(
                [
                    level_config["intercept"],  # Trend (intercept)
                    level_config["slope_base"] * (i % 2),  # Trend (slope)
                    level_config["phi_1"],  # phi_1 (AR)
                    level_config["theta_1"],  # theta_1 (MA)
                    level_config["Phi_1"],  # Phi_1 (Seasonal AR)
                    level_config["Theta_1"],  # Theta_1 (Seasonal MA)
                    level_config["variance"],  # Varianza de los residuos
                ]
            )

            demand = simulate_sarima(
                n_obs=n_obs,
                order=base_order,
                seasonal_order=base_seasonal_order,
                params=params,
                seed=int(level_config["base_seed"] + i),
            )
            all_series.append(demand)

    dates = pd.date_range(start="2022-01-01", periods=n_obs, freq="D")

    return pd.DataFrame(
        {
            "unique_id": np.repeat(np.arange(len(all_series)), n_obs).astype(
                str
            ),
            "ds": np.tile(dates, len(all_series)),
            "y": np.concatenate(all_series),
        }
    )


def championship(
    statsforecast: StatsForecast, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the championship of the models.

    Args:
        statsforecast: The statsforecast object.
        test_df: The test data frame.

    Returns:
        The championship data frame.
    """
    forecast = statsforecast.predict(h=30, level=[95])

    return cast(
        pd.DataFrame,
        (
            forecast.melt(
                id_vars=["unique_id", "ds"], var_name="model", value_name="yhat"
            )
            .assign(
                interval=lambda x: x.model.str.extract(".*?-(.*)$"),
                model=lambda x: x.model.str.replace("-.*$", "", regex=True),
            )
            .fillna("punctual")
            .pivot(
                index=["unique_id", "ds", "model"],
                columns="interval",
                values="yhat",
            )
            .reset_index()
            .merge(test_df, on=["unique_id", "ds"])
            .assign(
                winkler_score=lambda x: np.where(
                    x.y < x["lo-95"],
                    (x["hi-95"] - x["lo-95"]) + 2 / 0.95 * (x["lo-95"] - x.y),
                    np.where(
                        x.y > x["hi-95"],
                        (x["hi-95"] - x["lo-95"])
                        + 2 / 0.95 * (x.y - x["hi-95"]),
                        (x["hi-95"] - x["lo-95"]),
                    ),
                )
            )
            .groupby(["unique_id", "model"])
            .agg({"winkler_score": "mean"})
            .assign(
                rank_model=lambda x: x.groupby("unique_id").winkler_score.rank(
                    method="dense"
                )
            )
            .sort_values(["unique_id", "rank_model"])
            .loc[lambda x: x.rank_model == 1]
            .reset_index()
        ),
    )


def save_model(
    models: dict[str, _TS], filepath: str = "time_series.pkl"
) -> None:
    """
    Save the trained model to a pickle file.

    Args:
        model: Trained LinearRegression model
        filepath: Path where to save the model
    """
    with open(filepath, "wb") as f:
        pickle.dump(models, f)

    print(f"Model saved to {filepath}")


def load_model(
    filepath: str = "time_series.pkl",
) -> dict[str, _TS] | None:
    """
    Load a trained model from a pickle file.

    Args:
        filepath: Path to the model file

    Returns:
        dict[str, _TS] models or None if file doesn't exist
    """
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")

        return None

    with open(filepath, "rb") as f:
        model: dict[str, _TS] = pickle.load(f)

    print(f"Model loaded from {filepath}")

    return model


def predict(
        models: dict[str, _TS],
        h: int,
        level: list[int],
        unique_id: str) -> dict[str, list[float]]:
    """
    Make a prediction using the trained model.

    Args:
        models: Dictionary of trained _TS models
        h: The number of steps to predict
        level: Prediction interval levels calculated from conformal intervals
        unique_id: The unique id of the series

    Returns:
        Forecasted time series
    """
    model = models.get(unique_id, None)

    if not model:
        raise ValueError(f"Model not found for unique_id: {unique_id}")

    if h < 0:
        raise ValueError("h must be greater than 0")

    if h > OOT_SAMPLE_SIZE:
        raise ValueError(f"h must be less than {OOT_SAMPLE_SIZE}")

    prediction = model.predict(h=OOT_SAMPLE_SIZE, level=level)

    try:
        print(f"Prediction: {prediction}")
        return {k: p.tolist()[h:] for k, p in prediction.items()}
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise
