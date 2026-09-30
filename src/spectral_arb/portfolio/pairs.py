from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_hedge_ratio(
    log_prices_a: ArrayLike,
    log_prices_b: ArrayLike,
) -> tuple[float, float]:
    asset_a = np.asarray(log_prices_a, dtype=np.float64)
    asset_b = np.asarray(log_prices_b, dtype=np.float64)

    if asset_a.ndim != 1 or asset_b.ndim != 1:
        raise ValueError("both price inputs must be 1D arrays")
    if asset_a.shape != asset_b.shape:
        raise ValueError("price inputs must have the same length")
    if asset_a.size < 3:
        raise ValueError("at least three observations are required")
    if not np.isfinite(asset_a).all() or not np.isfinite(asset_b).all():
        raise ValueError("price inputs must contain only finite values")
    if np.var(asset_b) == 0:
        raise ValueError("asset_b log prices must have non-zero variance")

    design = np.column_stack([np.ones(asset_b.size), asset_b])
    intercept, hedge_ratio = np.linalg.lstsq(design, asset_a, rcond=None)[0]
    return float(intercept), float(hedge_ratio)


def compute_spread_levels(
    log_prices_a: ArrayLike,
    log_prices_b: ArrayLike,
    intercept: float,
    hedge_ratio: float,
) -> NDArray[np.float64]:
    asset_a = np.asarray(log_prices_a, dtype=np.float64)
    asset_b = np.asarray(log_prices_b, dtype=np.float64)

    if asset_a.ndim != 1 or asset_b.ndim != 1:
        raise ValueError("both price inputs must be 1D arrays")
    if asset_a.shape != asset_b.shape:
        raise ValueError("price inputs must have the same length")
    if not np.isfinite(asset_a).all() or not np.isfinite(asset_b).all():
        raise ValueError("price inputs must contain only finite values")
    if not np.isfinite(intercept) or not np.isfinite(hedge_ratio):
        raise ValueError("intercept and hedge_ratio must be finite")

    return asset_a - intercept - hedge_ratio * asset_b