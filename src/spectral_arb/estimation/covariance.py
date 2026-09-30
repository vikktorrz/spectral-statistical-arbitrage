"""Covariance estimation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_covariance(returns: ArrayLike) -> NDArray[np.float64]:
    """Compute the unbiased sample covariance matrix.

    Rows are observations; columns are assets.
    """
    values = np.asarray(returns, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("returns must be a 2D array shaped (observations, assets)")
    if values.shape[0] < 2:
        raise ValueError("returns must contain at least two observations")
    if values.shape[1] < 1:
        raise ValueError("returns must contain at least one asset")
    if not np.isfinite(values).all():
        raise ValueError("returns must contain only finite values")

    centered = values - values.mean(axis=0, keepdims=True)
    return centered.T @ centered / (values.shape[0] - 1)


def covariance_in_pair_space(
    asset_covariance: ArrayLike,
    exposure_matrix: ArrayLike,
) -> NDArray[np.float64]:
    """Transform asset covariance into covariance of pair-spread returns.

    If asset returns are r and pair exposures are A, pair returns are A.T @ r,
    so their covariance is A.T @ asset_covariance @ A.
    """
    covariance = np.asarray(asset_covariance, dtype=np.float64)
    exposures = np.asarray(exposure_matrix, dtype=np.float64)

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("asset_covariance must be a square matrix")
    if covariance.shape[0] == 0:
        raise ValueError("asset_covariance must contain at least one asset")
    if exposures.ndim != 2 or exposures.shape[0] != covariance.shape[0]:
        raise ValueError("exposure_matrix rows must match the asset covariance")
    if exposures.shape[1] == 0:
        raise ValueError("exposure_matrix must contain at least one pair")
    if not np.isfinite(covariance).all() or not np.isfinite(exposures).all():
        raise ValueError("inputs must contain only finite values")
    if not np.allclose(covariance, covariance.T, atol=1e-10):
        raise ValueError("asset_covariance must be symmetric")
    if np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("asset_covariance must be positive semidefinite")

    pair_covariance = exposures.T @ covariance @ exposures
    return 0.5 * (pair_covariance + pair_covariance.T)