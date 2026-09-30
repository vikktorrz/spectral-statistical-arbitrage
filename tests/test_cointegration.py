import numpy as np
import pytest

from spectral_arb.portfolio.cointegration import (
    benjamini_yekutieli,
    engle_granger_test,
)


def test_engle_granger_detects_a_constructed_cointegrated_pair():
    rng = np.random.default_rng(2026)
    log_price_b = np.cumsum(rng.normal(size=600))
    log_price_a = 1.2 * log_price_b + rng.normal(scale=0.15, size=600)

    statistic, p_value = engle_granger_test(log_price_a, log_price_b)

    assert np.isfinite(statistic)
    assert 0.0 <= p_value < 0.05


def test_benjamini_yekutieli_adjusts_pvalues_in_original_order():
    adjusted, selected = benjamini_yekutieli(
        [0.04, 0.001, 0.02],
        alpha=0.05,
    )

    np.testing.assert_allclose(adjusted, [0.22, 0.0055, 0.055])
    np.testing.assert_array_equal(selected, [False, True, False])


def test_benjamini_yekutieli_rejects_invalid_pvalues():
    with pytest.raises(ValueError, match="between 0 and 1"):
        benjamini_yekutieli([0.01, 1.2])