# Methodology

## Problem framing

Industrial demand forecasting in grocery retail is constrained by:
- **Decision frequency**: replenishment orders are placed weekly (fresh) or monthly (dry, non-food), so forecast accuracy at the *aggregated horizon* matters more than daily granularity.
- **Heterogeneous product behavior**: stable high-rotation SKUs vs intermittent long-tail vs promotion-driven products require different modeling assumptions.
- **Operational simplicity**: a model that's marginally better but 10× more complex to maintain may not be worth deploying.

This project takes these constraints seriously. The benchmark is designed to answer the question: **where does ML provide enough lift over Holt-Winters to justify the additional complexity?**

## Baseline: Holt-Winters multi-frequency

The reference is a Holt-Winters (triple exponential smoothing) baseline tuned per track:

- **Fresh (weekly)**: HW with weekly seasonality (period=52), damped trend, multiplicative seasonality
- **Dry (monthly)**: HW with annual seasonality (period=12), additive trend, additive seasonality
- **Non-food (monthly)**: HW with annual seasonality (period=12), damped trend

Implementation uses [`statsforecast`](https://nixtlaverse.nixtla.io/statsforecast) for vectorized fitting across thousands of series.

The baseline is intentionally well-tuned: a poor baseline would inflate ML gains and undermine the benchmark's credibility.

## Validation strategy

**Time series cross-validation with expanding window**:
- 4 folds, each holding out a contiguous test period
- Gap of 0 days between train and test (M5 has no leakage between rows)
- Metrics computed at the decision-relevant aggregation (weekly or monthly), not daily

**Why not random K-Fold?** Random shuffling leaks future information into training and produces optimistic metrics. Walk-forward validation is the only honest approach for time series.

## Metrics

Primary metrics are chosen for business relevance:

- **WMAPE** (Weighted Mean Absolute Percentage Error): handles zeros, weights by volume — proxies replenishment-cost-weighted error
- **Bias**: detects systematic over- or under-forecasting; critical for inventory policy
- **RMSE / MAE**: secondary, for completeness

**Segmented analysis** is the headline output:
- By rotation tier (top 20% / middle 60% / bottom 20% by volume)
- By promotion status (promoted vs non-promoted at forecast time)
- By product lifecycle stage (new, mature, declining)
- By intermittence (regular vs intermittent demand pattern)

This segmented view replaces the misleading single-number comparison.

## Feature engineering rationale

Each feature family is motivated by a specific Holt-Winters limitation:

| HW limitation | ML feature family | Expected segments to gain |
|---|---|---|
| No exogenous variables | Promo features (price drops, discount depth, promo duration) | Promoted SKUs |
| No multi-seasonality | Calendar features (DOW × month interactions, holidays, SNAP days) | All segments |
| No information transfer | Hierarchical aggregates (category mean, store mean, etc.) | New / low-volume SKUs |
| Stockout sensitivity | Stockout detection + imputation | SKUs with frequent OOS |
| Single seasonal pattern | Lags at multiple scales (D-1, D-7, D-28, D-365) | Products with dual seasonality |
| Lifecycle-blind | Product age, lifecycle phase indicators | New products |

## Hierarchical reconciliation

In production, forecasts at different aggregation levels (SKU × store, SKU × region, SKU × warehouse, total) often disagree. Reconciliation methods enforce coherence:

- **Bottom-up**: forecast at the lowest level, sum up
- **Top-down**: forecast at the top, allocate using historical proportions
- **MinT (Minimum Trace)**: optimal reconciliation using forecast error covariance ([Wickramasuriya et al., 2019](https://robjhyndman.com/papers/MinT.pdf))

This project compares all three at the **warehouse aggregation level**, which is the level driving supplier replenishment in real-world retail.

## Reproducibility

- All random seeds are pinned (see `forecasting.config.RANDOM_SEED`)
- Environment is locked via `uv.lock`
- Data download is scripted (`scripts/download_data.py`)
- Each experiment is tracked with MLflow, parameters and metrics logged
- Configurations are stored as YAML files under `configs/`
