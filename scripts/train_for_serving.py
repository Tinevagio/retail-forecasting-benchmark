"""Train a model on the full track history and save it for API serving.

This script is the bridge between the experimental scripts (which evaluate
models on multiple folds) and the API (which serves a single trained model).

For each model we want to serve, this script:
1. Loads the data
2. Builds the full feature set
3. Trains the model on ALL available history (no test holdout — we've
   already done that in the benchmarks)
4. Saves the trained Forecaster to artifacts/<name>.pkl
5. Saves the latest features per series to artifacts/<name>_features.parquet

Usage:
    uv run python scripts/train_for_serving.py
    uv run python scripts/train_for_serving.py --models lightgbm-tweedie holt-winters
    uv run python scripts/train_for_serving.py --sample-stores CA_1
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

from forecasting.data.aggregate import TRACK_CONFIGS, prepare_track
from forecasting.data.load import load_calendar, load_prices, load_sales, melt_sales
from forecasting.features.event_features import (
    NAMED_EVENTS_OF_INTEREST,
    add_weekly_event_features,
    add_weekly_snap_features,
)
from forecasting.features.hierarchical_features import (
    add_hierarchical_lag_features,
    build_id_to_hierarchy,
)
from forecasting.features.lags import (
    add_basic_time_features,
    add_lag_features,
    add_rolling_features,
)
from forecasting.features.promo_track import add_promo_features_weekly
from forecasting.models.base import Forecaster
from forecasting.models.holt_winters import HoltWintersBaseline
from forecasting.models.lightgbm_model import LightGBMForecaster
from forecasting.serving.feature_store import save_feature_store
from forecasting.serving.persistence import ModelArtifact, save_artifact

LAGS = [1, 2, 4, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26]
ROLLING_STATS = ("mean", "std")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", default="fresh_weekly", choices=list(TRACK_CONFIGS))
    p.add_argument(
        "--models",
        nargs="+",
        default=["lightgbm-tweedie"],
        help="Models to train: lightgbm-tweedie, holt-winters",
    )
    p.add_argument(
        "--sample-stores",
        nargs="*",
        default=None,
        help="Restrict to specific stores for fast iteration.",
    )
    p.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Output directory for artifacts and feature stores.",
    )
    return p.parse_args()


def build_features(
    track_data: pl.DataFrame,
    daily_sales: pl.DataFrame,
    calendar: pl.DataFrame,
    prices: pl.DataFrame,
    track_id_format: str = "item_store",
) -> tuple[pl.DataFrame, list[str]]:
    """Build the same feature set used in phase 3.3."""
    df = track_data
    df = add_lag_features(df, lags=LAGS)
    df = add_rolling_features(df, windows=ROLLING_WINDOWS, statistics=ROLLING_STATS)
    df = add_basic_time_features(df)
    df = add_weekly_snap_features(df, calendar, daily_sales.select(["id", "state_id"]).unique())
    df = add_weekly_event_features(df, calendar)
    id_to_hier = build_id_to_hierarchy(daily_sales, track_id_format)
    df = add_hierarchical_lag_features(
        df,
        id_to_hier,
        lag=1,
        levels=("dept_id", "store_id", "cat_id"),
    )
    df = add_promo_features_weekly(df, prices, calendar, id_to_hier, track_id_format)

    feature_cols = (
        [f"sales_lag_{k}" for k in LAGS]
        + [f"sales_roll{w}_{s}" for w in ROLLING_WINDOWS for s in ROLLING_STATS]
        + ["month", "week_of_year", "year", "day_of_year"]
        + ["snap_days_in_week"]
        + [f"event_{ev}_in_week" for ev in NAMED_EVENTS_OF_INTEREST]
        + ["event_other_in_week", "any_event_in_week"]
        + ["dept_id_avg_sales_lag_1", "store_id_avg_sales_lag_1", "cat_id_avg_sales_lag_1"]
        + ["is_on_promo", "price_relative_to_ref"]
    )
    return df, feature_cols


def main() -> int:
    args = parse_args()
    cfg = TRACK_CONFIGS[args.track]
    artifacts_dir = Path(args.artifacts_dir)

    print(f"=== Training for serving: {args.track} ===")
    print(f"Models requested: {', '.join(args.models)}")
    print(f"Output: {artifacts_dir}/")

    # 1. Load data
    print("\n[1/4] Loading data...")
    t0 = time.time()
    calendar = load_calendar()
    sales_wide = load_sales()
    prices = load_prices()
    sales = melt_sales(sales_wide, calendar=calendar)
    if args.sample_stores:
        sales = sales.filter(pl.col("store_id").is_in(args.sample_stores))
    print(f"     Loaded in {time.time() - t0:.1f}s")

    # 2. Prepare track + features
    print(f"\n[2/4] Preparing track {args.track}...")
    track_data = prepare_track(sales, args.track)
    track_id_format = "item_store" if cfg["level"] == "sku_store" else "item_state"
    df, feature_cols = build_features(
        track_data,
        sales,
        calendar,
        prices,
        track_id_format,
    )
    print(f"     Track shape: {df.shape}, {len(feature_cols)} features")
    train_end_val = df["date"].max()
    assert isinstance(train_end_val, date), (
        f"Expected date for train_end, got {type(train_end_val).__name__}"
    )
    train_end = train_end_val
    print(f"     Train end: {train_end}")

    # 3. Train each model
    print("\n[3/4] Training models...")
    horizon = cfg["horizon_periods"]
    models_to_save: list[tuple[str, Forecaster]] = []

    for model_key in args.models:
        # Annotated as the abstract Forecaster type so mypy accepts both
        # LightGBMForecaster and HoltWintersBaseline assignments
        model: Forecaster

        if model_key == "lightgbm-tweedie":
            print(f"\n  > {model_key}: training LightGBM (Tweedie + correction)...")
            t1 = time.time()
            model = LightGBMForecaster(
                feature_cols=feature_cols,
                horizon=horizon,
                frequency=cfg["frequency"],
                n_estimators=200,
                objective="tweedie",
                target_correction="rolling_mean",
            )
            model.fit(df)
            print(f"     Trained in {time.time() - t1:.1f}s")
            models_to_save.append((model_key, model))

        elif model_key == "holt-winters":
            print(f"\n  > {model_key}: training Holt-Winters (slow)...")
            t1 = time.time()
            # HW doesn't use the engineered features; trains on raw track
            model = HoltWintersBaseline(
                season_length=cfg["seasonality"],
                frequency=cfg["frequency"],
                n_jobs=1,
            )
            model.fit(track_data)
            print(f"     Trained in {time.time() - t1:.1f}s")
            models_to_save.append((model_key, model))

        else:
            print(f"  ! Unknown model {model_key!r}, skipping")

    # 4. Persist
    print("\n[4/4] Persisting artifacts...")
    for model_key, trained_model in models_to_save:
        artifact_path = artifacts_dir / f"{model_key}.pkl"
        features_path = artifacts_dir / f"{model_key}_features.parquet"

        artifact = ModelArtifact(
            model=trained_model,
            model_name=model_key,
            train_end=train_end,
            feature_cols=feature_cols if model_key.startswith("lightgbm") else [],
            track_name=args.track,
            extra={"horizon": horizon, "frequency": cfg["frequency"]},
        )
        save_artifact(artifact, artifact_path)

        # For HW, the feature store is just (id, date, sales) since the model
        # uses internal state. For LightGBM, it's the engineered features.
        if model_key.startswith("lightgbm"):
            save_feature_store(df, features_path, feature_cols=feature_cols)
        else:
            save_feature_store(track_data, features_path, feature_cols=["sales"])

        print(f"     {model_key}: {artifact_path} + {features_path}")

    print("\nDone. To serve, run:")
    print("  uv run uvicorn forecasting.serving.api:app --reload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
