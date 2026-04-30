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


## Phase 2.2 — Holt-Winters baseline on fresh_weekly track

### Setup

- **Track**: fresh_weekly (FOODS_3 SKU x store, weekly aggregation, ~24,000 series)
- **Cross-validation**: 4-fold walk-forward, 4-week test horizon per fold
- **Test windows**: Jan 26 - May 16, 2016 (4 contiguous 4-week periods)
- **Models**: 3 naive baselines + Holt-Winters via statsforecast.AutoETS
  with cold-start fallback to DriftNaive (min_history = 2 * season_length = 104 weeks)
- **Total predictions evaluated**: 131,680 per model (4 folds x 4 weeks x ~8200 series)
- **Compute time**: 7,800 seconds (2h10) for the full run, single-threaded

### Headline results

| Model | WMAPE (mean ± std) | Bias | RMSE | MAE |
|-------|-------------------|------|------|-----|
| **HoltWinters (AutoETS)** | **0.343 ± 0.012** | -5.3% | 13.7 | 5.7 |
| DriftNaive | 0.380 ± 0.015 | -6.6% | 14.5 | 6.3 |
| HistoricalMean | 0.521 ± 0.017 | -13.9% | 18.8 | 8.7 |
| SeasonalNaive(52) | 0.608 ± 0.033 | -15.0% | 23.7 | 10.1 |

**Headline gain**: Holt-Winters reduces WMAPE by 9.5% relative to DriftNaive
(0.343 vs 0.380), at the cost of ~50x longer training time. The improvement
is consistent across folds (low std) and confirmed on a 100-SKU sanity-check
sample (gain of 9.4% in that controlled setting).

### Holt-Winters coverage

- **Total series**: 8,230 (FOODS_3 SKU x store, where FOODS_3 contains
  ~823 unique items in 10 stores)
- **Handled by AutoETS** (>= 104 weeks history): 8,230 (100%)
- **Fallback to DriftNaive**: 0 (0%)

The dataset has enough history that no series triggered the cold-start
fallback. The headline HW WMAPE of 0.343 reflects pure AutoETS predictions,
not a blend with DriftNaive. This makes the comparison with DriftNaive
(0.380) a clean apples-to-apples test of statistical method.

Note: the smaller series count (8,230) compared to the full M5 dataset
(30,490) reflects the track's category filter (FOODS_3 only). The other
two tracks (dry_monthly, non_food_monthly) will cover the remaining
categories in subsequent runs.

### Key observations

1. **Holt-Winters delivers a modest but real gain.** The 9.5% relative
   improvement on WMAPE is statistically solid (low across-fold variance)
   and consistent between sanity-check and full-run scales. Holt-Winters
   is a worthy baseline to beat for ML.

2. **Bias also improves.** HoltWinters has the lowest forecast bias (-5.3%)
   among the four models, beating even DriftNaive (-6.6%). For supply-chain
   contexts where bias drives stockout cascades (see EDA, takeaway 5), this
   is an underrated benefit.

3. **SeasonalNaive(52) is decisively the worst.** This empirically confirms
   the EDA finding that annual seasonality on FOODS is weak (1.09x amplitude
   ratio). AutoETS' AIC search likely selects non-seasonal ETS variants
   for most series in this category.

4. **The test window matters.** The 4-fold evaluation covers Jan-May 2016,
   which excludes the strong-seasonality periods (Christmas, back-to-school,
   summer). A wider evaluation period might favor HoltWinters more on
   seasonally-driven products. This is a known limitation of the current
   benchmark.

### Floor to beat for ML

The next phase (LightGBM with engineered features) must improve on:
- WMAPE: 0.343 (Holt-Winters)
- Bias: -5.3% (Holt-Winters)

Expected ML edge over Holt-Winters: SNAP days (+10% effect), named events
(per-event encoding), promotional weeks (+46-81% lift), cold-start products,
and stockout-corrected training data — none of which Holt-Winters can leverage.

The bar is set: any ML model on this track must beat 0.343 WMAPE to justify
its complexity.
