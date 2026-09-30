from __future__ import annotations

import pandas as pd

from spectral_arb.portfolio.cointegration import (
    benjamini_yekutieli,
    engle_granger_test,
)
from spectral_arb.portfolio.pairs import estimate_hedge_ratio


def scan_cointegrated_pairs(
    log_prices: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test every asset pair in a formation window.

    The returned table is sorted by adjusted p-value. Use only the formation
    period here; do not include the later trading or evaluation period.
    """
    if not isinstance(log_prices, pd.DataFrame):
        raise ValueError("log_prices must be a pandas DataFrame")
    if log_prices.shape[0] < 20:
        raise ValueError("at least 20 observations are required")
    if log_prices.shape[1] < 2:
        raise ValueError("at least two assets are required")
    if log_prices.columns.has_duplicates:
        raise ValueError("asset names must be unique")
    if not all(isinstance(name, str) for name in log_prices.columns):
        raise ValueError("asset names must be strings")
    if not log_prices.map(lambda value: pd.notna(value)).all().all():
        raise ValueError("log_prices must not contain missing values")

    # A fixed name ordering makes the Engle-Granger regression direction
    # reproducible, independent of the DataFrame's input column order.
    names = sorted(log_prices.columns)
    records: list[dict[str, float | str]] = []

    for index, asset_a in enumerate(names):
        for asset_b in names[index + 1 :]:
            series_a = log_prices[asset_a].to_numpy(dtype=float)
            series_b = log_prices[asset_b].to_numpy(dtype=float)

            intercept, hedge_ratio = estimate_hedge_ratio(series_a, series_b)
            statistic, p_value = engle_granger_test(series_a, series_b)

            records.append(
                {
                    "asset_a": asset_a,
                    "asset_b": asset_b,
                    "intercept": intercept,
                    "hedge_ratio": hedge_ratio,
                    "test_statistic": statistic,
                    "p_value": p_value,
                }
            )

    adjusted, selected = benjamini_yekutieli(
        [record["p_value"] for record in records],
        alpha=alpha,
    )
    for record, adjusted_p_value, is_selected in zip(
        records, adjusted, selected, strict=True
    ):
        record["adjusted_p_value"] = float(adjusted_p_value)
        record["selected"] = bool(is_selected)

    return pd.DataFrame(records).sort_values(
        "adjusted_p_value",
        kind="mergesort",
    ).reset_index(drop=True)