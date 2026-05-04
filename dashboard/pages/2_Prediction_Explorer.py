"""Prediction Explorer page.

The interactive demo: pick a SKU, see its history and the model's forecast
on a single chart. This is the page to show during interviews — it makes
the abstract benchmark numbers tangible.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

from forecasting.serving.feature_store import (
    load_feature_store,
)
from forecasting.serving.persistence import load_artifact

st.set_page_config(page_title="Prediction Explorer", page_icon="🔮", layout="wide")

st.title("🔮 Prediction Explorer")
st.caption("Pick a SKU and a model, see its history and forecast.")


# ---- Data loading (cached) -----------------------------------------------

ARTIFACTS_DIR = Path("artifacts")
DATA_PATH = Path("data/processed/track_fresh_weekly.parquet")


@st.cache_resource
def list_available_models() -> list[str]:
    """Find all .pkl files paired with a feature store in artifacts/."""
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in ARTIFACTS_DIR.glob("*.pkl")
        if (ARTIFACTS_DIR / f"{p.stem}_features.parquet").exists()
    )


@st.cache_resource
def load_model(model_name: str):
    """Load a model artifact (cached as a long-lived resource)."""
    artifact = load_artifact(ARTIFACTS_DIR / f"{model_name}.pkl")
    feature_store = load_feature_store(ARTIFACTS_DIR / f"{model_name}_features.parquet")
    return artifact, feature_store


@st.cache_data
def load_track_history() -> pl.DataFrame | None:
    """Load the historical sales for the fresh_weekly track.

    Tries a pre-computed parquet first; otherwise rebuilds from raw data.
    Rebuilding is slow (~30s) but only happens once thanks to caching.
    """
    if DATA_PATH.exists():
        return pl.read_parquet(DATA_PATH)

    # Fallback: rebuild from raw data
    try:
        from forecasting.data.aggregate import prepare_track
        from forecasting.data.load import load_calendar, load_sales, melt_sales

        calendar = load_calendar()
        sales = melt_sales(load_sales(), calendar=calendar)
        track = prepare_track(sales, "fresh_weekly")
        # Cache to disk for next time
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        track.write_parquet(DATA_PATH)
        return track
    except (FileNotFoundError, ImportError):
        return None


# ---- Sidebar: model and SKU selection ------------------------------------

models = list_available_models()
if not models:
    st.error(
        "No models found in `artifacts/`. Run\n\n"
        "```bash\nuv run python scripts/train_for_serving.py --sample-stores CA_1\n```\n\n"
        "to generate them."
    )
    st.stop()

with st.sidebar:
    st.header("Selection")
    model_name = st.selectbox("Model", options=models, index=0)
    artifact, feature_store = load_model(model_name)

    available_ids = sorted(feature_store["id"].unique().to_list())
    st.caption(f"{len(available_ids):,} SKUs available")

    # Default to the first SKU; could add filtering by store/dept later
    series_id = st.selectbox(
        "SKU x store",
        options=available_ids,
        index=0,
        help="Series identifier (format: ITEM__STORE)",
    )

    horizon = st.slider("Forecast horizon (weeks)", 1, 12, 4)

    show_other_models = st.checkbox("Compare with other models", value=len(models) > 1)


# ---- Main: forecast and history ------------------------------------------

st.subheader(f"Forecast for `{series_id}`")
st.caption(
    f"Model: **{artifact.model_name}** · "
    f"Trained on data up to **{artifact.train_end}** · "
    f"Horizon: **{horizon} weeks**"
)


# Get prediction from the selected model
predictions_df = artifact.model.predict(horizon=horizon, ids=[series_id])

# Try to get historical sales
history_df = load_track_history()
if history_df is None:
    st.warning("Couldn't load historical sales (raw data not available). Showing forecast only.")
    history_pd = pd.DataFrame()
else:
    sku_history = history_df.filter(pl.col("id") == series_id).sort("date")
    history_pd = sku_history.select(["date", "sales"]).to_pandas()
    # Restrict to the last 2 years for readability
    if len(history_pd) > 0:
        cutoff = pd.Timestamp(artifact.train_end - timedelta(days=730))
        history_pd = history_pd[history_pd["date"] >= cutoff]


# ---- Build the chart ------------------------------------------------------

# Combine history + prediction in a single long-format DF for st.line_chart
chart_rows = []
for _, row in history_pd.iterrows():
    chart_rows.append(
        {
            "date": pd.Timestamp(row["date"]),
            "value": row["sales"],
            "series": "Historical sales",
        }
    )

for row in predictions_df.iter_rows(named=True):
    chart_rows.append(
        {
            "date": pd.Timestamp(row["date"]),
            "value": row["prediction"],
            "series": f"Forecast ({artifact.model_name})",
        }
    )

# If asked, run the other models on the same SKU
if show_other_models:
    for other_name in models:
        if other_name == model_name:
            continue
        try:
            other_artifact, _ = load_model(other_name)
            other_preds = other_artifact.model.predict(horizon=horizon, ids=[series_id])
            for row in other_preds.iter_rows(named=True):
                chart_rows.append(
                    {
                        "date": pd.Timestamp(row["date"]),
                        "value": row["prediction"],
                        "series": f"Forecast ({other_artifact.model_name})",
                    }
                )
        except (FileNotFoundError, ValueError, KeyError):
            continue

if not chart_rows:
    st.warning("No data to display.")
    st.stop()

chart_df = pd.DataFrame(chart_rows)
chart_pivot = chart_df.pivot_table(index="date", columns="series", values="value", aggfunc="first")
st.line_chart(chart_pivot, width="stretch")


# ---- Numeric forecast table ----------------------------------------------

st.subheader("Predicted values")
preds_display = predictions_df.to_pandas()
preds_display["prediction"] = preds_display["prediction"].round(2)
preds_display.columns = ["Series", "Date", "Predicted sales"]
st.dataframe(preds_display, width="stretch", hide_index=True)


# ---- Stats panel ---------------------------------------------------------

if not history_pd.empty:
    st.divider()
    st.subheader("Series stats")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Periods of history", len(sku_history))
    with col2:
        st.metric("Mean weekly sales", f"{sku_history['sales'].mean():.1f}")
    with col3:
        st.metric("Max weekly sales", f"{sku_history['sales'].max():.0f}")
    with col4:
        zero_share = (sku_history["sales"] == 0).sum() / len(sku_history)
        st.metric("Zero-sales share", f"{100 * zero_share:.1f}%")
