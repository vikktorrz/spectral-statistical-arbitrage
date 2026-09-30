from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_covariance(returns: ArrayLike) -> NDArray[np.float64]:
    """Compute the unbiased sample covariance matrix.

    Parameters
    ----------
    returns:
        A finite 2D array shaped (observations, assets).

    Returns
    -------
    A covariance matrix shaped (assets, assets).
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