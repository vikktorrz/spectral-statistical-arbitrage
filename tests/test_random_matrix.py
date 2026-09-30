import numpy as np
import pytest

from spectral_arb.estimation.random_matrix import (
    marchenko_pastur_bounds,
    rmt_clean_covariance,
)


def test_marchenko_pastur_bounds():
    lower, upper = marchenko_pastur_bounds(
        n_observations=100,
        n_assets=25,
    )

    assert lower == pytest.approx(0.25)
    assert upper == pytest.approx(2.25)


def test_rmt_clean_covariance_is_symmetric_and_positive_semidefinite():
    rng = np.random.default_rng(42)
    returns = rng.normal(size=(300, 12))

    covariance = rmt_clean_covariance(returns)

    assert covariance.shape == (12, 12)
    np.testing.assert_allclose(covariance, covariance.T, atol=1e-10)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10


def test_rmt_clean_covariance_preserves_sample_variances():
    rng = np.random.default_rng(7)
    returns = rng.normal(size=(250, 5)) * np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    covariance = rmt_clean_covariance(returns)
    sample_variances = returns.var(axis=0, ddof=1)

    np.testing.assert_allclose(np.diag(covariance), sample_variances, rtol=1e-10)


def test_rmt_clean_covariance_rejects_constant_asset():
    returns = np.column_stack([np.arange(10.0), np.ones(10)])

    with pytest.raises(ValueError, match="non-zero"):
        rmt_clean_covariance(returns)