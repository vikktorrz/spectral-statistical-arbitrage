from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def mean_reversion_forecast(
    spread_history: ArrayLike,
    lookback: int = 60,
    expected_return_per_zscore: float = 0.001,
) -> NDArray[np.float64]:
    values = np.asarray(spread_history, dtype=np.float64)

    if values.ndim != 2:
        raise ValueError("spread_history must be a 2D time-by-spread array")
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if values.shape[0] < lookback + 1:
        raise ValueError("spread_history needs lookback observations plus the latest value")
    if values.shape[1] < 1:
        raise ValueError("spread_history must contain at least one spread")
    if not np.isfinite(values).all():
        raise ValueError("spread_history must contain only finite values")
    if expected_return_per_zscore < 0:
        raise ValueError("expected_return_per_zscore cannot be negative")

    reference = values[-lookback - 1 : -1]
    latest = values[-1]
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0, ddof=1)

    if np.any(scale == 0):
        raise ValueError("each spread must have non-zero historical variance")

    z_scores = (latest - mean) / scale
    return -expected_return_per_zscore * z_scores