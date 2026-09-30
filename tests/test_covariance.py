import numpy as np
import pytest

from spectral_arb.estimation.covariance import sample_covariance


def test_sample_covariance_values_and_shape():
    returns = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    covariance = sample_covariance(returns)

    assert covariance.shape == (2, 2)
    np.testing.assert_allclose(covariance, [[4.0, 4.0], [4.0, 4.0]])


def test_sample_covariance_one_asset_keeps_matrix_shape():
    covariance = sample_covariance(np.array([[1.0], [2.0], [3.0]]))

    assert covariance.shape == (1, 1)
    np.testing.assert_allclose(covariance, [[1.0]])


def test_sample_covariance_rejects_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        sample_covariance(np.array([[1.0, np.nan], [2.0, 3.0]]))


def test_sample_covariance_requires_two_dimensions():
    with pytest.raises(ValueError, match="2D"):
        sample_covariance(np.array([1.0, 2.0, 3.0]))