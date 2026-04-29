# Results

> ⚠️ This document will be progressively populated as the project advances. Phases below correspond to the project roadmap.

## Phase 1 — Holt-Winters baseline

*To be completed at end of phase 1.*

Expected content:
- Baseline WMAPE per track and per segment
- Diagnostic of Holt-Winters' weak spots (where it fails)
- Sanity checks vs naive baselines

## Phase 2 — LightGBM with engineered features

*To be completed at end of phase 2.*

Expected content:
- Headline lift vs Holt-Winters, per track
- Segmented analysis (promo vs non-promo, top vs tail, new vs mature)
- Feature importance and ablation studies
- Hyperparameter tuning results

## Phase 3 — Deep learning and hierarchical reconciliation

*To be completed at end of phase 3.*

Expected content:
- N-BEATS / TFT comparison vs LightGBM
- Reconciliation methods comparison (bottom-up, top-down, MinT)
- Decision-level metrics at warehouse aggregation

## Phase 4 — Industrialization

*To be completed at end of phase 4.*

Expected content:
- API performance and latency benchmarks
- Dashboard screenshots
- Deployment notes

## Limitations and honest caveats

This section will document the project's known limitations and the things I'd address with more time / resources.


## Phase 2.1 — Naive baselines on fresh_weekly track

**Setup**: 4-fold walk-forward CV, 4-week test horizon per fold, on the
fresh_weekly track (FOODS_3 SKU x store, weekly aggregation).

| Model | WMAPE (mean ± std) | Bias | RMSE | MAE |
|-------|-------------------|------|------|-----|
| DriftNaive | **0.380 ± 0.015** | -6.6% | 14.5 | 6.3 |
| HistoricalMean | 0.521 ± 0.017 | -13.9% | 18.8 | 8.7 |
| SeasonalNaive(52) | 0.608 ± 0.033 | -15.0% | 23.7 | 10.1 |

**Key observation**: SeasonalNaive(52) underperforms HistoricalMean,
consistent with the EDA finding that annual seasonality on FOODS is weak
(amplitude ratio 1.09x). On this track, copying last year's value injects
noise rather than signal. DriftNaive wins by capturing recent level + trend.

**Floor to beat**: Holt-Winters and ML models must clear WMAPE = 0.380 to
justify their complexity on this track.

**Caveat**: the test window covers Jan-May 2016, which excludes major
seasonal events. A wider evaluation period may favor seasonal models more.
