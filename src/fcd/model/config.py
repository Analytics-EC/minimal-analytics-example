from statsforecast.models import (  # type: ignore
    MSTL,
    AutoARIMA,
    AutoCES,
    AutoETS,
    AutoTheta,
)
from statsforecast.utils import ConformalIntervals  # type: ignore

NOBS = 365 * 3
BASE_ORDER = (1, 0, 1)
BASE_SEASONAL_ORDER = (1, 0, 1, 7)
OOT_SAMPLE_SIZE = 30

intervals = ConformalIntervals(h=30, n_windows=2)
MODELS = [
    AutoARIMA(
        season_length=7,
        allowdrift=True,
        allowmean=True,
        seasonal=True,
        max_p=2,
        max_q=2,
        max_P=2,
        max_Q=2,
        max_d=2,
        max_D=2,
        prediction_intervals=intervals,
    ),
    AutoCES(season_length=7, prediction_intervals=intervals),
    AutoTheta(season_length=7, prediction_intervals=intervals),
    AutoETS(season_length=7, prediction_intervals=intervals),
    MSTL(season_length=[7, 365], prediction_intervals=intervals),
]

DEMAND_LEVELS: dict[str, dict[str, float]] = {
    "very_low": {
        "count": 2,
        "base_seed": 42,
        "intercept": 25.0,
        "slope_base": 0.01,
        "phi_1": 0.8,
        "theta_1": 0.5,
        "Phi_1": 0.5,
        "Theta_1": 0.3,
        "variance": 50.0,
    },
    "low": {
        "count": 3,
        "base_seed": 50,
        "intercept": 50.0,
        "slope_base": 0.02,
        "phi_1": 0.6,
        "theta_1": 0.4,
        "Phi_1": 0.5,
        "Theta_1": 0.3,
        "variance": 50.0,
    },
    "medium": {
        "count": 3,
        "base_seed": 60,
        "intercept": 100.0,
        "slope_base": 0.03,
        "phi_1": 0.4,
        "theta_1": 0.3,
        "Phi_1": 0.5,
        "Theta_1": 0.3,
        "variance": 50.0,
    },
    "high": {
        "count": 2,
        "base_seed": 70,
        "intercept": 200.0,
        "slope_base": 0.03,
        "phi_1": 0.3,
        "theta_1": 0.2,
        "Phi_1": 0.5,
        "Theta_1": 0.3,
        "variance": 100.0,
    },
}
