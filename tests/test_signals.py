import numpy as np
import pytest

from spectral_arb.portfolio.signals import mean_reversion_forecast


def test_high_spread_level_produces_negative_forecast():
    history = np.array([[1.0], [2.0], [3.0], [5.0]])

    forecast = mean_reversion_forecast(
        history,
        lookback=3,
        expected_return_per_zscore=0.001,
    )

    np.testing.assert_allclose(forecast, [-0.003])


def test_low_spread_level_produces_positive_forecast():
    history = np.array([[1.0], [2.0], [3.0], [-1.0]])

    forecast = mean_reversion_forecast(
        history,
        lookback=3,
        expected_return_per_zscore=0.001,
    )

    np.testing.assert_allclose(forecast, [0.003])


def test_signal_rejects_constant_reference_spread():
    history = np.array([[2.0], [2.0], [2.0], [3.0]])

    with pytest.raises(ValueError, match="non-zero"):
        mean_reversion_forecast(history, lookback=3)