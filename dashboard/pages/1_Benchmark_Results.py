"""Benchmark Results page.

Interactive table and charts of the model evaluation. Reads the per-fold
results from data/results/phase33_per_fold.csv if available, otherwise
falls back to the headline numbers documented in the project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Benchmark Results", page_icon="📈", layout="wide")

st.title("📈 Benchmark Results")
st.caption("Holt-Winters and LightGBM variants on the fresh_weekly track")


# ---- Data loading --------------------------------------------------------


@st.cache_data
def load_per_fold_results() -> pd.DataFrame | None:
    """Load per-fold results from disk if available."""
    csv_path = Path("data/results/phase33_per_fold.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_feature_importance() -> pd.DataFrame | None:
    """Load LightGBM feature importance from disk if available."""
    csv_path = Path("data/results/phase33_feature_importance.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# Static fallback data — the headline numbers from the documented run
SUMMARY_FALLBACK = pd.DataFrame(
    {
        "model_name": [
            "HoltWinters (AutoETS)",
            "LightGBM tweedie + correction",
            "LightGBM tweedie",
            "LightGBM regression_l1",
            "DriftNaive",
        ],
        "wmape_mean": [0.343, 0.360, 0.363, 0.357, 0.380],
        "wmape_std": [0.012, 0.014, 0.019, 0.013, 0.015],
        "bias_mean": [-0.053, -0.029, -0.057, -0.109, -0.066],
        "rmse_mean": [13.7, 14.1, 14.4, 14.6, 14.5],
        "mae_mean": [5.7, 6.0, 6.0, 5.9, 6.3],
        "n_obs_total": [131680] * 5,
    }
)


# ---- Summary table -------------------------------------------------------

st.header("Headline summary")
st.caption(
    "WMAPE = Weighted MAPE (lower is better). Bias = "
    "mean(prediction - actual) / mean(|actual|) — closer to 0 is better."
)

per_fold = load_per_fold_results()
if per_fold is not None:
    # Compute summary on the fly from per-fold data
    summary = (
        per_fold.groupby("model_name")
        .agg(
            wmape_mean=("wmape", "mean"),
            wmape_std=("wmape", "std"),
            bias_mean=("bias", "mean"),
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            n_obs_total=("n_obs", "sum"),
        )
        .reset_index()
        .sort_values("wmape_mean")
    )
    st.success("Showing live data from data/results/phase33_per_fold.csv")
else:
    summary = SUMMARY_FALLBACK
    st.info(
        "Showing static fallback data. Run "
        "`uv run python scripts/run_lightgbm_phase33.py` to populate "
        "live results."
    )

# Format for display
display = summary.copy()
display["WMAPE"] = display["wmape_mean"].apply(lambda x: f"{x:.3f}")
if "wmape_std" in display.columns:
    display["WMAPE std"] = display["wmape_std"].apply(lambda x: f"{x:.3f}")
display["Bias"] = display["bias_mean"].apply(lambda x: f"{100 * x:+.1f}%")
display["RMSE"] = display["rmse_mean"].apply(lambda x: f"{x:.1f}")
display["MAE"] = display["mae_mean"].apply(lambda x: f"{x:.1f}")
display["N obs"] = display["n_obs_total"].apply(lambda x: f"{int(x):,}")

st.dataframe(
    display[["model_name", "WMAPE", "WMAPE std", "Bias", "RMSE", "MAE", "N obs"]]
    if "WMAPE std" in display.columns
    else display[["model_name", "WMAPE", "Bias", "RMSE", "MAE", "N obs"]],
    use_container_width=True,
    hide_index=True,
)


# ---- WMAPE chart ---------------------------------------------------------

st.header("WMAPE comparison")
chart_data = summary.set_index("model_name")[["wmape_mean"]].rename(columns={"wmape_mean": "WMAPE"})
st.bar_chart(chart_data, horizontal=True)
st.caption("Lower is better. Holt-Winters wins by a 4-5% margin on this track.")


# ---- Bias chart ----------------------------------------------------------

st.header("Bias comparison")
bias_chart = summary.set_index("model_name")[["bias_mean"]].rename(columns={"bias_mean": "Bias"})
st.bar_chart(bias_chart, horizontal=True)
st.caption(
    "Closer to zero is better. The Tweedie + correction variant achieves the "
    "lowest absolute bias (-2.9%) — better than Holt-Winters on this metric."
)


# ---- Per-fold detail (only if real data is loaded) -----------------------

if per_fold is not None:
    st.header("Per-fold breakdown")
    st.caption("Walk-forward folds, oldest first.")
    pivoted = per_fold.pivot_table(index="fold_id", columns="model_name", values="wmape")
    st.line_chart(pivoted)
    st.dataframe(per_fold, use_container_width=True, hide_index=True)


# ---- Feature importance ---------------------------------------------------

st.header("LightGBM feature importance")
st.caption(
    "Average gain across the 4 horizon-specific models (Tweedie + correction "
    "variant, last fold's training data)."
)

importance = load_feature_importance()
if importance is not None:
    top_n = st.slider("Top N features", 5, 30, 15)
    top = importance.head(top_n).set_index("feature")[["importance"]]
    st.bar_chart(top, horizontal=True)
    st.dataframe(importance.head(top_n), use_container_width=True, hide_index=True)
else:
    st.info(
        "Feature importance file not found. "
        "Run `uv run python scripts/run_lightgbm_phase33.py` to generate it."
    )


# ---- Methodology note ----------------------------------------------------

st.divider()
st.subheader("Methodology")

st.markdown(
    """
- **Cross-validation**: 4-fold walk-forward, 4-week test horizon per fold,
  testing on Jan-May 2016.
- **Track**: `fresh_weekly` (FOODS_3 SKU x store, weekly aggregation, 8,230 series).
- **HoltWinters fallback**: series shorter than 104 weeks (2 x season_length)
  fall back to DriftNaive. On the full track, 100% of series have enough
  history for AutoETS.
- **LightGBM features**: 31 columns including lags (1, 2, 4, 8, 13, 26, 52
  weeks), rolling means/stds (4, 13, 26 windows), calendar features, SNAP
  days, named events, hierarchical lags (dept/store/cat), and promo flags.
- **Stockout correction**: applied only to TRAINING data (test target left
  as-is). Suspicious zeros (zero following non-zero rolling mean) are
  imputed with rolling-mean values.
"""
)
