import numpy as np
from statsmodels.tsa.statespace import sarimax as sm  # type: ignore


def simulate_sarima(
    n_obs: int,
    order: tuple,
    seasonal_order: tuple,
    params: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simula una serie de tiempo SARIMAX con los parámetros especificados.

    Args:
        n_obs (int): Número de observaciones a simular.
        order (tuple): Tupla (p, d, q) del modelo no estacional.
        seasonal_order (tuple): Tupla (P, D, Q, s) del modelo estacional.
        params (np.ndarray): Array con los parámetros del modelo.
        seed (int): Semilla para la reproducibilidad.

    Returns:
        np.ndarray: Un array de NumPy con la serie de tiempo simulada.
    """
    sarima_model = sm.SARIMAX(
        endog=np.random.rand(n_obs),
        order=order,
        seasonal_order=seasonal_order,
        trend="ct",
    )

    simulated_series = sarima_model.simulate(
        params=params, nsimulations=n_obs, anchor="start", random_state=seed
    )

    return simulated_series.flatten()  # type: ignore
