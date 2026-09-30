from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize


def optimize_portfolio(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    previous_weights: ArrayLike | None = None,
    risk_aversion: float = 1.0,
    transaction_cost: float = 0.0,
    max_gross_leverage: float = 1.0,
    max_position_weight: float = 0.05,
    market_neutral: bool = True,
) -> NDArray[np.float64]:
    mu = np.asarray(expected_returns, dtype=np.float64)
    sigma = np.asarray(covariance, dtype=np.float64)

    if mu.ndim != 1:
        raise ValueError("expected_returns must be a 1D vector")
    n_assets = mu.size
    if n_assets == 0:
        raise ValueError("at least one asset is required")
    if sigma.shape != (n_assets, n_assets):
        raise ValueError("covariance shape must match expected_returns")
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all():
        raise ValueError("inputs must contain only finite values")
    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(sigma).min() < -1e-8:
        raise ValueError("covariance must be positive semidefinite")
    if risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive")
    if transaction_cost < 0:
        raise ValueError("transaction_cost cannot be negative")
    if max_gross_leverage <= 0 or max_position_weight <= 0:
        raise ValueError("leverage and position limits must be positive")

    if previous_weights is None:
        previous = np.zeros(n_assets)
    else:
        previous = np.asarray(previous_weights, dtype=np.float64)
        if previous.shape != (n_assets,) or not np.isfinite(previous).all():
            raise ValueError("previous_weights must be a finite vector matching the assets")

    n_variables = 3 * n_assets
    weights_slice = slice(0, n_assets)
    abs_weights_slice = slice(n_assets, 2 * n_assets)
    turnover_slice = slice(2 * n_assets, 3 * n_assets)

    matrix = np.zeros((4 * n_assets + 1, n_variables))
    for i in range(n_assets):
        matrix[i, i] = -1.0
        matrix[i, n_assets + i] = 1.0

        matrix[n_assets + i, i] = 1.0
        matrix[n_assets + i, n_assets + i] = 1.0

        matrix[2 * n_assets + i, i] = -1.0
        matrix[2 * n_assets + i, 2 * n_assets + i] = 1.0

        matrix[3 * n_assets + i, i] = 1.0
        matrix[3 * n_assets + i, 2 * n_assets + i] = 1.0

    matrix[-1, abs_weights_slice] = 1.0

    lower = np.concatenate(
        [
            np.zeros(2 * n_assets),
            -previous,
            previous,
            [-np.inf],
        ]
    )
    upper = np.concatenate([np.full(4 * n_assets, np.inf), [max_gross_leverage]])
    constraints = [LinearConstraint(matrix, lower, upper)]

    if market_neutral:
        net_exposure = np.zeros((1, n_variables))
        net_exposure[0, weights_slice] = 1.0
        constraints.append(LinearConstraint(net_exposure, [0.0], [0.0]))

    bounds = Bounds(
        np.concatenate(
            [
                np.full(n_assets, -max_position_weight),
                np.zeros(2 * n_assets),
            ]
        ),
        np.concatenate(
            [
                np.full(n_assets, max_position_weight),
                np.full(n_assets, max_position_weight),
                np.full(n_assets, np.inf),
            ]
        ),
    )

    initial = np.concatenate([np.zeros(n_assets), np.zeros(n_assets), np.abs(previous)])

    def objective(variables: NDArray[np.float64]) -> float:
        weights = variables[weights_slice]
        turnover = variables[turnover_slice]
        return (
            0.5 * risk_aversion * weights @ sigma @ weights
            - mu @ weights
            + transaction_cost * turnover.sum()
        )

    def gradient(variables: NDArray[np.float64]) -> NDArray[np.float64]:
        weights = variables[weights_slice]
        return np.concatenate(
            [
                risk_aversion * sigma @ weights - mu,
                np.zeros(n_assets),
                np.full(n_assets, transaction_cost),
            ]
        )

    result = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"portfolio optimization failed: {result.message}")

    return result.x[weights_slice]