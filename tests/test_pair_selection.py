import numpy as np
import pandas as pd

from spectral_arb.portfolio.selection import scan_cointegrated_pairs


def test_scanner_finds_a_strong_formation_window_pair():
    rng = np.random.default_rng(13)
    common_trend = np.cumsum(rng.normal(size=500))

    prices = pd.DataFrame(
        {
            "A": 1.7 * common_trend + rng.normal(scale=0.1, size=500),
            "B": common_trend,
            "C": np.cumsum(rng.normal(size=500)),
        }
    )

    results = scan_cointegrated_pairs(prices)
    selected = set(
        zip(
            results.loc[results["selected"], "asset_a"],
            results.loc[results["selected"], "asset_b"],
            strict=True,
        )
    )

    assert len(results) == 3
    assert ("A", "B") in selected