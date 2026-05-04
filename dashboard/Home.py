"""Home page of the Streamlit dashboard.

This is the first thing visitors see. The goal is to communicate, in 30
seconds:
- what the project does
- the headline benchmark result
- how to navigate to the deeper pages
"""

from __future__ import annotations

import streamlit as st

# ---- Page config ----------------------------------------------------------

st.set_page_config(
    page_title="Retail Forecasting Benchmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---- Header ---------------------------------------------------------------

st.title("📊 Retail Forecasting Benchmark")
st.markdown("*A reproducible comparison of statistical and ML methods on the M5 Walmart dataset.*")


# ---- Project pitch in 3 columns ------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🎯 What")
    st.write(
        "An end-to-end forecasting pipeline benchmarking Holt-Winters "
        "(AutoETS) against LightGBM with EDA-motivated features, on "
        "30,490 retail series over 5 years of history."
    )

with col2:
    st.subheader("🏗️ How")
    st.write(
        "Walk-forward cross-validation, identical splits and metrics for "
        "all models, 150 unit & integration tests, FastAPI service, "
        "Docker-ready deployment."
    )

with col3:
    st.subheader("🔍 Why")
    st.write(
        "An honest benchmark to interrogate when ML actually beats well-"
        "tuned classical methods in retail forecasting — and where the "
        "tradeoffs really lie."
    )

st.divider()


# ---- Headline results ----------------------------------------------------

st.header("Headline results — fresh_weekly track")
st.caption("8,230 SKU x store series, 4-fold walk-forward CV, 131,680 predictions per model")

results_data = {
    "Model": [
        "**Holt-Winters (AutoETS)**",
        "LightGBM Tweedie + stockout correction",
        "LightGBM Tweedie",
        "LightGBM regression_l1",
        "DriftNaive",
    ],
    "WMAPE": [0.343, 0.360, 0.363, 0.357, 0.380],
    "Bias": ["-5.3%", "**-2.9%**", "-5.7%", "-10.9%", "-6.6%"],
    "Notes": [
        "Champion on this track and window.",
        "Best ML variant. Bias divided by ~4.",
        "Tweedie alone halves the bias.",
        "L1 baseline before bias correction.",
        "Floor: linear extrapolation.",
    ],
}

st.table(results_data)


# ---- Key takeaways -------------------------------------------------------

st.subheader("Key findings")

st.markdown(
    """
1. **Holt-Winters retains the lead** on this track and test window. Well-tuned
   classical methods remain competitive on stable retail series.

2. **Tweedie loss + stockout correction reduce LightGBM's bias by 73%**
   (from -10.9% to -2.9%) — the most operationally valuable improvement in
   contexts where bias drives stockouts.

3. **Feature importance is dominated by rolling statistics** of the target.
   EDA-motivated features (SNAP, events, hierarchical lags) under-perform
   expectations, partly because the test window (Jan-May 2016) excludes
   major events like Christmas and Thanksgiving.

4. **The bar to beat for ML on retail forecasting** is higher than the
   community often assumes. Beating Holt-Winters is achievable but requires
   genuinely informative features — not just model complexity.
"""
)

st.divider()


# ---- Architecture --------------------------------------------------------

st.header("Architecture")

st.markdown(
    """
```
┌─────────────────────────────────────────────────────────────┐
│                 retail-forecasting-benchmark                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   data/        ──>  Load M5 + features                      │
│   models/      ──>  Holt-Winters, LightGBM, naive baselines │
│   evaluation/  ──>  Walk-forward CV + metrics               │
│   serving/     ──>  FastAPI app + persistence layer         │
│                                                             │
│   ┌──────────────────────────────────────────────┐          │
│   │  Train script ──>  artifacts/  ──>  API      │          │
│   │       │                                       │          │
│   │       └──>  feature store (.parquet)          │          │
│   └──────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Quality gates** : 150 tests passing, 88% coverage, ruff strict, mypy strict, GitHub Actions CI green.
"""
)


# ---- Navigation ----------------------------------------------------------

st.divider()
st.header("Explore further")

col_nav1, col_nav2 = st.columns(2)

with col_nav1:
    st.markdown(
        """
### 📈 Benchmark Results
Interactive view of all model results with per-fold breakdowns,
metrics comparison, and feature importance.
"""
    )

with col_nav2:
    st.markdown(
        """
### 🔮 Prediction Explorer
Pick a SKU and a model, see the historical sales and the model's
4-week forecast, with side-by-side comparison.
"""
    )

st.caption("👈 Use the sidebar to navigate.")


# ---- Footer --------------------------------------------------------------

st.divider()
st.caption(
    "Built as a portfolio project for ML Engineer reconversion. "
    "Source: [GitHub](https://github.com/Tinevagio/retail-forecasting-benchmark)."
)
