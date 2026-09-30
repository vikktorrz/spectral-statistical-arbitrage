from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from statsmodels.tsa.stattools import coint


def engle_granger_test(
    log_prices_a: ArrayLike,
    log_prices_b: ArrayLike,
    maxlag: int | None = None,
) -> tuple[float, float]:
    asset_a = np.asarray(log_prices_a, dtype=np.float64)
    asset_b = np.asarray(log_prices_b, dtype=np.float64)

    if asset_a.ndim != 1 or asset_b.ndim != 1:
        raise ValueError("both log-price inputs must be 1D arrays")
    if asset_a.shape != asset_b.shape:
        raise ValueError("log-price inputs must have the same length")
    if asset_a.size < 20:
        raise ValueError("at least 20 observations are required")
    if not np.isfinite(asset_a).all() or not np.isfinite(asset_b).all():
        raise ValueError("log-price inputs must contain only finite values")
    if np.var(asset_a) == 0 or np.var(asset_b) == 0:
        raise ValueError("both log-price series must have non-zero variance")
    if maxlag is not None and maxlag < 0:
        raise ValueError("maxlag cannot be negative")

    statistic, p_value, _ = coint(
        asset_a,
        asset_b,
        trend="c",
        maxlag=maxlag,
        autolag="aic",
    )
    return float(statistic), float(p_value)


def benjamini_yekutieli(
    p_values: ArrayLike,
    alpha: float = 0.05,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    values = np.asarray(p_values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("p_values must be a 1D array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be finite and between 0 and 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    n_tests = values.size
    if n_tests == 0:
        return values.copy(), np.zeros(0, dtype=bool)

    order = np.argsort(values)
    sorted_p = values[order]
    ranks = np.arange(1, n_tests + 1)
    dependence_factor = np.sum(1.0 / ranks)

    adjusted_sorted = sorted_p * n_tests * dependence_factor / ranks
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(n_tests, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return adjusted, adjusted <= alpha