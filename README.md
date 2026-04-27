# Retail Forecasting Benchmark

> Benchmarking ML approaches against industrial Holt-Winters baselines for multi-level retail demand forecasting.

[![CI](https://github.com/Tinevagio/retail-forecasting-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/Tinevagio/retail-forecasting-benchmark/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Context

Industrial demand forecasting tools in grocery retail (e.g., AZAP, RELEX, ToolsGroup) typically rely on **Holt-Winters exponential smoothing** with sector-specific frequencies: weekly forecasts for fresh products, monthly for dry goods and non-food. While robust, these methods have well-known limitations:

- **No exogenous variables**: promotions, calendar events, and weather are ignored
- **No information transfer**: each SKU is treated in isolation (cold-start blind)
- **Limited multi-seasonality**: single seasonal component
- **Sensitivity to stockouts**: zeros from lost sales pollute the historical signal

This project quantifies *where* and *by how much* modern ML approaches improve upon a well-tuned Holt-Winters baseline, on the [M5 Forecasting](https://www.kaggle.com/competitions/m5-forecasting-accuracy) dataset (Walmart, ~30k SKUs × 10 stores).

## Approach

The benchmark is structured around three forecasting tracks reflecting real-world retail decision frequencies:

| Track | Aggregation | Horizon | Mapped M5 categories |
|-------|-------------|---------|----------------------|
| Fresh | Weekly | 1-4 weeks | FOODS_3 (fast-moving) |
| Dry / Grocery | Monthly | 1-3 months | FOODS_1, FOODS_2, HOUSEHOLD |
| Non-food | Monthly | 1-3 months | HOBBIES |

Each track is evaluated at the **decision-relevant aggregation** (weekly or monthly totals), not at daily granularity, mirroring how forecast quality actually drives replenishment.

## Models compared

- **Naive baselines**: seasonal naive, moving average
- **Holt-Winters** (multi-frequency, mimics AZAP): the industrial baseline to beat
- **LightGBM**: ML approach with engineered features targeting Holt-Winters blind spots
- **Deep Learning** (N-BEATS / TFT via [Darts](https://github.com/unit8co/darts)): comparison on rich segments
- **Hierarchical reconciliation** (MinT, bottom-up, top-down via [HierarchicalForecast](https://nixtlaverse.nixtla.io/hierarchicalforecast))

## Key findings

> *Results to be populated as the project progresses. Expected highlights:*
> - Promotion-sensitive products: ML improves WMAPE by 25-40%
> - Stable high-rotation products: marginal gains, suggesting hybrid routing
> - New products (cold-start): ML dominates via hierarchical feature transfer

## Repository structure

```
src/forecasting/        # Core Python package
├── data/              # Loading, preprocessing, splits
├── features/          # Feature engineering modules
├── models/            # Model implementations
├── evaluation/        # Metrics and segmented analysis
├── reconciliation/    # Hierarchical forecast reconciliation
└── pipelines/         # Training and prediction pipelines

notebooks/             # Exploratory analyses (read-only artifacts)
scripts/               # CLI entry points
tests/                 # Unit tests
configs/               # YAML configs for experiments
docs/                  # Methodology and results documentation
```

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (fast Python package manager)
- Kaggle API credentials (to download M5 data)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/retail-forecasting-benchmark.git
cd retail-forecasting-benchmark

# Install dependencies
uv sync

# Download M5 data (requires Kaggle API setup)
make data

# Run tests
make test

# Train Holt-Winters baseline
make baseline

# Train LightGBM model
make train-lgbm

# Run full evaluation
make evaluate
```

## Methodology

Detailed methodology, including feature engineering rationale, validation strategy (walk-forward time series CV with gap), and metric definitions (WMAPE weighted by sales, bias, segmented analysis) is documented in [`docs/methodology.md`](docs/methodology.md).

## Results

Full results, segmented analysis, and discussion of trade-offs are in [`docs/results.md`](docs/results.md).

## About this project

I work in supply chain / forecasting in grocery retail and built this project to systematically evaluate where ML provides real value over well-established statistical methods. The goal is not to claim ML wins everywhere — it doesn't — but to identify the conditions under which the additional complexity is justified.

## License

MIT
