from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def marchenko_pastur_bounds(
    n_observations: int,
    n_assets: int,
    variance: float = 1.0,
) -> tuple[float, float]:
    if n_observations <= 0 or n_assets <= 0:
        raise ValueError("n_observations and n_assets must be positive")
    if variance <= 0:
        raise ValueError("variance must be positive")

    q = n_assets / n_observations
    root_q = np.sqrt(q)
    lower = variance * (1.0 - root_q) ** 2
    upper = variance * (1.0 + root_q) ** 2
    return lower, upper


def rmt_clean_covariance(
    returns: ArrayLike,
    preserve_top_components: int = 0,
) -> NDArray[np.float64]:
    values = np.asarray(returns, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("returns must be a 2D array shaped (observations, assets)")
    n_observations, n_assets = values.shape
    if n_observations < 2:
        raise ValueError("returns must contain at least two observations")
    if n_assets < 1:
        raise ValueError("returns must contain at least one asset")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")
    if not 0 <= preserve_top_components <= n_assets:
        raise ValueError("preserve_top_components must be between 0 and n_assets")

    centered = values - values.mean(axis=0, keepdims=True)
    volatilities = centered.std(axis=0, ddof=1)
    if np.any(volatilities == 0):
        raise ValueError("each asset must have non-zero return variance")

    standardized = centered / volatilities
    correlation = standardized.T @ standardized / (n_observations - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    _, upper_edge = marchenko_pastur_bounds(n_observations, n_assets)

    noise_mask = eigenvalues <= upper_edge
    if preserve_top_components:
        noise_mask[-preserve_top_components:] = False

    if noise_mask.any():
        eigenvalues[noise_mask] = eigenvalues[noise_mask].mean()

    cleaned = (eigenvectors * eigenvalues) @ eigenvectors.T

    # Restore unit diagonal after spectral filtering.
    diagonal_scale = np.sqrt(np.clip(np.diag(cleaned), 1e-12, None))
    cleaned = cleaned / np.outer(diagonal_scale, diagonal_scale)

    # Restore each asset's sample volatility.
    return cleaned * np.outer(volatilities, volatilities)