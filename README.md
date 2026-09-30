# Spectral Statistical Arbitrage

A reproducible research project investigating whether random matrix theory (RMT)
covariance cleaning improves portfolio construction and statistical arbitrage
performance out of sample.

## Research question

Does RMT-based covariance regularization produce more stable portfolios than
conventional covariance estimators after transaction costs and realistic constraints?

This repository is a research project, not an investment product. No performance
claims should be made until experiments are completed and their assumptions documented.

## Planned workflow

1. Define the asset universe, data sources, and point-in-time rules.
2. Estimate sample and RMT-cleaned covariance matrices on rolling windows.
3. Construct portfolio signals and risk-aware weights.
4. Backtest with turnover, transaction costs, and position constraints.
5. Compare against documented baselines using out-of-sample results.

## Repository structure

- `src/spectral_arb/` — reusable Python package
- `configs/` — version-controlled experiment settings
- `experiments/` — notebooks and research entry points
- `scripts/` — command-line runners
- `tests/` — package tests
- `reports/` — generated summaries; outputs are not committed

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Reproducibility and data

Record each experiment's configuration, date range, universe, data source, random
seed, estimator parameters, and cost assumptions. Keep raw or licensed market data
outside Git. Avoid look-ahead and survivorship bias by documenting point-in-time
universe membership and data availability.

## License

MIT for the code. This does not grant rights to market data; follow each data
provider's terms.