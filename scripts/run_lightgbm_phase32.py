"""Phase 3.2: LightGBM with EDA-motivated features.

Compared to phase 3.1, we add:
- Per-week SNAP day count (state-aware)
- Per-named-event flags (Christmas, Thanksgiving, Pesach End, etc.)
- Hierarchical lag features (dept/store mean at lag 1)
- Promo features (is_on_promo + price_relative_to_ref)

Holt-Winters is included only via --include-hw (slow, optional).

Usage:
    uv run python scripts/run_lightgbm_phase32.py --sample-stores CA_1 --sample-skus 100
    uv run python scripts/run_lightgbm_phase32.py  # full track
"""

from __future__ import annotations

import argparse
import sys
import time

import polars as pl

from forecasting.data.aggregate import TRACK_CONFIGS, prepare_track
from forecasting.data.load import load_calendar, load_prices, load_sales, melt_sales
from forecasting.data.splits import make_walk_forward_folds
from forecasting.evaluation.runner import aggregate_by_fold, evaluate_models
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
from forecasting.models.holt_winters import HoltWintersBaseline
from forecasting.models.lightgbm_model import LightGBMForecaster
from forecasting.models.naive import DriftNaive

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
        "--include-hw",
        action="store_true",
        help="Include Holt-Winters (slow ~2h on full track).",
    )
    return p.parse_args()


def build_features(
    track_data: pl.DataFrame,
    daily_sales: pl.DataFrame,
    calendar: pl.DataFrame,
    prices: pl.DataFrame,
    track_id_format: str = "item_store",
) -> tuple[pl.DataFrame, list[str]]:
    """Build the full phase 3.2 feature set."""
    df = track_data

    # 1. Lag and rolling features (phase 3.1)
    df = add_lag_features(df, lags=LAGS)
    df = add_rolling_features(df, windows=ROLLING_WINDOWS, statistics=ROLLING_STATS)
    df = add_basic_time_features(df)

    # 2. SNAP features (state-aware, weekly aggregated)
    df = add_weekly_snap_features(df, calendar, daily_sales.select(["id", "state_id"]).unique())

    # 3. Named event features
    df = add_weekly_event_features(df, calendar)

    # 4. Hierarchical lag features
    id_to_hier = build_id_to_hierarchy(daily_sales, track_id_format)
    df = add_hierarchical_lag_features(
        df,
        id_to_hier,
        lag=1,
        levels=("dept_id", "store_id", "cat_id"),
    )

    # 5. Promo features
    df = add_promo_features_weekly(df, prices, calendar, id_to_hier, track_id_format)

    # Build the feature column list
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

    print(f"=== Phase 3.2: LightGBM (full features) on {args.track} ===")

    t0 = time.time()
    print("\n[1/6] Loading data...")
    calendar = load_calendar()
    sales_wide = load_sales()
    prices = load_prices()
    sales = melt_sales(sales_wide, calendar=calendar)
    print(f"     Loaded in {time.time() - t0:.1f}s. Sales: {sales.shape}")

    if args.sample_stores:
        sales = sales.filter(pl.col("store_id").is_in(args.sample_stores))
        print(f"     Filtered to stores {args.sample_stores}: {sales.shape}")

    print(f"\n[2/6] Preparing track {args.track}...")
    track_data = prepare_track(sales, args.track)
    print(f"     Track data: {track_data.shape}")

    if args.sample_skus:
        sample_ids = (
            track_data["id"]
            .unique()
            .sort()  # determinism: unique() doesn't guarantee order
            .sample(args.sample_skus, seed=42)
            .to_list()
        )
        track_data = track_data.filter(pl.col("id").is_in(sample_ids))
        print(f"     Sampled {args.sample_skus} SKUs: {track_data.shape}")

    print("\n[3/6] Building features...")
    t1 = time.time()
    track_id_format = "item_store" if cfg["level"] == "sku_store" else "item_state"
    df, feature_cols = build_features(track_data, sales, calendar, prices, track_id_format)
    print(f"     Features built in {time.time() - t1:.1f}s. Total: {len(feature_cols)} features")
    print(f"     Final shape: {df.shape}")

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
    if args.include_hw:
        models.append(
            HoltWintersBaseline(
                season_length=cfg["seasonality"],
                frequency=cfg["frequency"],
                n_jobs=1,
            )
        )

    t2 = time.time()
    results = evaluate_models(models, df, folds, horizon=horizon)
    print(f"     Evaluation completed in {time.time() - t2:.1f}s")

    print("\n[6/6] Summary (sorted by WMAPE):")
    summary = aggregate_by_fold(results)
    print(summary)

    # Feature importance
    print("\n=== LightGBM feature importance (from last fold) ===")
    last_fold = folds[-1]
    train_last = df.filter(pl.col("date") <= pl.lit(last_fold.train_end))
    lgb_demo = LightGBMForecaster(
        feature_cols=feature_cols,
        horizon=horizon,
        frequency=cfg["frequency"],
        n_estimators=200,
    )
    lgb_demo.fit(train_last)
    importance = lgb_demo.feature_importance()
    print(importance.head(20))

    return 0


if __name__ == "__main__":
    sys.exit(main())
