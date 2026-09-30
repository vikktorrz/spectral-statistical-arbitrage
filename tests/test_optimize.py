import numpy as np
import pytest

from spectral_arb.portfolio.optimize import optimize_portfolio


def test_optimizer_respects_market_neutrality_and_exposure_limits():
    expected_returns = np.array([0.02, -0.01, 0.01, -0.02])
    covariance = np.eye(4) * 0.01

    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        max_gross_leverage=0.8,
        max_position_weight=0.3,
        market_neutral=True,
    )

    assert weights.sum() == pytest.approx(0.0, abs=1e-7)
    assert np.abs(weights).sum() <= 0.8 + 1e-7
    assert np.abs(weights).max() <= 0.3 + 1e-7


def test_optimizer_accounts_for_previous_weights():
    expected_returns = np.array([0.01, -0.01])
    covariance = np.eye(2) * 0.01
    previous_weights = np.array([0.1, -0.1])

    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        previous_weights=previous_weights,
        transaction_cost=0.001,
        max_gross_leverage=0.5,
        max_position_weight=0.3,
    )

    assert weights.sum() == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(weights).all()


def test_optimizer_rejects_non_symmetric_covariance():
    with pytest.raises(ValueError, match="symmetric"):
        optimize_portfolio(
            expected_returns=[0.01, -0.01],
            covariance=[[1.0, 0.2], [0.1, 1.0]],
        )