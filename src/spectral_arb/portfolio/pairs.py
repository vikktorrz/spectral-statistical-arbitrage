"""Pair-spread construction and hedge-ratio estimation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def estimate_hedge_ratio(
    log_prices_a: ArrayLike,
    log_prices_b: ArrayLike,
) -> tuple[float, float]:
    """Fit log(price_a) = intercept + hedge_ratio * log(price_b).

    Use only a formation/training window when estimating these parameters.
    """
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
    """Compute log(A) - intercept - hedge_ratio * log(B)."""
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


def build_pair_exposure_matrix(
    asset_names: Sequence[str],
    pairs: Sequence[tuple[str, str, float]],
) -> NDArray[np.float64]:
    """Map pair positions to asset positions.

    Each pair is (asset_a, asset_b, hedge_ratio), matching the spread
    definition log(A) - hedge_ratio * log(B).
    """
    names = list(asset_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("asset_names must be non-empty and unique")
    if not pairs:
        raise ValueError("at least one pair is required")

    asset_index = {name: index for index, name in enumerate(names)}
    exposures = np.zeros((len(names), len(pairs)), dtype=np.float64)
    seen_pairs: set[tuple[str, str]] = set()

    for column, (asset_a, asset_b, hedge_ratio) in enumerate(pairs):
        if asset_a == asset_b:
            raise ValueError("a pair must contain two different assets")
        if asset_a not in asset_index or asset_b not in asset_index:
            raise ValueError("every pair asset must exist in asset_names")
        if not np.isfinite(hedge_ratio) or hedge_ratio == 0:
            raise ValueError("hedge_ratio must be finite and non-zero")

        pair_key = tuple(sorted((asset_a, asset_b)))
        if pair_key in seen_pairs:
            raise ValueError("duplicate pair")
        seen_pairs.add(pair_key)

        exposures[asset_index[asset_a], column] = 1.0
        exposures[asset_index[asset_b], column] = -hedge_ratio

    return exposures