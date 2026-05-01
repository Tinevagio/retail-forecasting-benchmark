"""Phase 3.1: LightGBM with basic features (lags + rolling + simple calendar).

This is the entry point for the ML phase. We compare a basic LightGBM
forecaster against the established Holt-Winters baseline (WMAPE 0.343).

The feature set is intentionally minimal at this stage:
- Lag features at 1, 2, 4, 8, 13, 26, 52 weeks
- Rolling mean and std at 4, 13, 26 weeks
- Basic calendar (month, week_of_year, year, day_of_year)

Phase 3.2 will add the EDA-motivated features (per-event encoding, SNAP,
hierarchical aggregates, promo features). Phase 3.3 will tune
hyperparameters.

Usage:
    uv run python scripts/run_lightgbm_phase31.py
    uv run python scripts/run_lightgbm_phase31.py --sample-stores CA_1 --sample-skus 100
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from forecasting.data.aggregate import TRACK_CONFIGS, prepare_track
from forecasting.data.load import load_calendar, load_sales, melt_sales
from forecasting.data.splits import make_walk_forward_folds
from forecasting.evaluation.runner import aggregate_by_fold, evaluate_models
from forecasting.features.lags import (
    add_basic_time_features,
    add_lag_features,
    add_rolling_features,
)
from forecasting.models.holt_winters import HoltWintersBaseline
from forecasting.models.lightgbm_model import LightGBMForecaster
from forecasting.models.naive import DriftNaive

# Feature configuration for the fresh_weekly track
LAGS = [1, 2, 4, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26]
ROLLING_STATS = ("mean", "std")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--track", default="fresh_weekly", choices=list(TRACK_CONFIGS))
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--sample-stores", nargs="*", default=None)
    p.add_argument("--sample-skus", type=int, default=None)
    p.add_argument(
        "--skip-hw",
        action="store_true",
        help="Skip Holt-Winters (already evaluated in phase 2.2). Saves ~2h on full run.",
    )
    return p.parse_args()


def build_features(track_data: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Add lag, rolling, and time features. Return enriched df + feature list."""
    df = track_data
    df = add_lag_features(df, lags=LAGS)
    df = add_rolling_features(df, windows=ROLLING_WINDOWS, statistics=ROLLING_STATS)
    df = add_basic_time_features(df)

    feature_cols = (
        [f"sales_lag_{k}" for k in LAGS]
        + [f"sales_roll{w}_{s}" for w in ROLLING_WINDOWS for s in ROLLING_STATS]
        + ["month", "week_of_year", "year", "day_of_year"]
    )
    return df, feature_cols


def main() -> int:
    args = parse_args()
    cfg = TRACK_CONFIGS[args.track]

    print(f"=== Phase 3.1: LightGBM on {args.track} ===")

    # 1. Load
    t0 = time.time()
    print("\n[1/6] Loading data...")
    calendar = load_calendar()
    sales_wide = load_sales()
    sales = melt_sales(sales_wide, calendar=calendar)
    print(f"     Loaded in {time.time() - t0:.1f}s. Shape: {sales.shape}")

    if args.sample_stores:
        sales = sales.filter(pl.col("store_id").is_in(args.sample_stores))
        print(f"     Filtered to stores {args.sample_stores}: {sales.shape}")

    # 2. Prepare track
    print(f"\n[2/6] Preparing track {args.track}...")
    track_data = prepare_track(sales, args.track)
    print(f"     Track data: {track_data.shape}")

    if args.sample_skus:
        sample_ids = track_data["id"].unique().sample(args.sample_skus, seed=42).to_list()
        track_data = track_data.filter(pl.col("id").is_in(sample_ids))
        print(f"     Sampled {args.sample_skus} SKUs: {track_data.shape}")

    # 3. Feature engineering
    print("\n[3/6] Building features...")
    t1 = time.time()
    track_with_features, feature_cols = build_features(track_data)
    print(f"     Features built in {time.time() - t1:.1f}s")
    print(f"     Total feature columns: {len(feature_cols)}")
    print(f"     Sample features: {feature_cols[:5]}...")

    # 4. Folds
    print(f"\n[4/6] Building {args.n_folds} walk-forward folds...")
    horizon = cfg["horizon_periods"]
    horizon_days = horizon * (7 if cfg["frequency"] == "W" else 30)
    folds = make_walk_forward_folds(
        min_date=track_data["date"].min(),
        max_date=track_data["date"].max(),
        n_folds=args.n_folds,
        test_horizon_days=horizon_days,
    )
    for f in folds:
        print(f"     {f}")

    # 5. Models
    print("\n[5/6] Fitting models...")
    models: list = [
        DriftNaive(frequency=cfg["frequency"]),
        LightGBMForecaster(
            feature_cols=feature_cols,
            horizon=horizon,
            frequency=cfg["frequency"],
            n_estimators=200,
        ),
    ]
    if not args.skip_hw:
        models.append(
            HoltWintersBaseline(
                season_length=cfg["seasonality"],
                frequency=cfg["frequency"],
                n_jobs=1,
            )
        )

    t2 = time.time()
    results = evaluate_models(models, track_with_features, folds, horizon=horizon)
    print(f"     Evaluation completed in {time.time() - t2:.1f}s")

    # 6. Summary
    print("\n[6/6] Summary (sorted by WMAPE):")
    summary = aggregate_by_fold(results)
    print(summary)

    # Feature importance for LightGBM (refit on the last fold's training data)
    print("\n=== LightGBM feature importance (from last fold) ===")
    last_fold = folds[-1]
    train_last = track_with_features.filter(pl.col("date") <= pl.lit(last_fold.train_end))
    lgb_demo = LightGBMForecaster(
        feature_cols=feature_cols,
        horizon=horizon,
        frequency=cfg["frequency"],
        n_estimators=200,
    )
    lgb_demo.fit(train_last)
    importance = lgb_demo.feature_importance()
    print(importance.head(15))

    return 0


if __name__ == "__main__":
    sys.exit(main())
