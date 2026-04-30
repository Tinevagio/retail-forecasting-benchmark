"""Run Holt-Winters baseline against naive baselines on the fresh_weekly track.

Usage:
    uv run python scripts/run_holt_winters.py

Optional flag for fast iteration on a sample:
    uv run python scripts/run_holt_winters.py --sample-stores CA_1 --sample-skus 100

Note on runtime:
    On the full fresh_weekly track (~24k series, 4 folds), expect 5-15 min
    depending on machine. Use --sample-stores for quick iteration during
    development.
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
from forecasting.models.holt_winters import HoltWintersBaseline
from forecasting.models.naive import DriftNaive, HistoricalMean, SeasonalNaive


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--track",
        default="fresh_weekly",
        choices=list(TRACK_CONFIGS),
        help="Which track to evaluate.",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of walk-forward folds.",
    )
    p.add_argument(
        "--sample-stores",
        nargs="*",
        default=None,
        help="Restrict to a subset of stores for fast iteration (e.g. --sample-stores CA_1 CA_3).",
    )
    p.add_argument(
        "--sample-skus",
        type=int,
        default=None,
        help="Random sample of N SKUs for fast iteration.",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for AutoETS. Use 1 on Windows for stability.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TRACK_CONFIGS[args.track]

    print(f"=== {args.track} ===")
    print(
        f"Frequency: {cfg['frequency']}, season_length: {cfg['seasonality']}, "
        f"horizon: {cfg['horizon_periods']} periods"
    )

    # 1. Load data
    t0 = time.time()
    print("\n[1/5] Loading data...")
    calendar = load_calendar()
    sales_wide = load_sales()
    sales = melt_sales(sales_wide, calendar=calendar)
    print(f"     Loaded in {time.time() - t0:.1f}s. Shape: {sales.shape}")

    # Optional subsetting
    if args.sample_stores:
        sales = sales.filter(pl.col("store_id").is_in(args.sample_stores))
        print(f"     Filtered to stores {args.sample_stores}: {sales.shape}")

    # 2. Prepare track (filter + aggregate)
    print(f"\n[2/5] Preparing track {args.track}...")
    track_data = prepare_track(sales, args.track)
    print(f"     Track data: {track_data.shape}")

    if args.sample_skus:
        sample_ids = track_data["id"].unique().sample(args.sample_skus, seed=42).to_list()
        track_data = track_data.filter(pl.col("id").is_in(sample_ids))
        print(f"     Sampled {args.sample_skus} SKUs: {track_data.shape}")

    # 3. Build folds
    print(f"\n[3/5] Building {args.n_folds} walk-forward folds...")
    horizon_periods = cfg["horizon_periods"]
    horizon_days = horizon_periods * (7 if cfg["frequency"] == "W" else 30)
    folds = make_walk_forward_folds(
        min_date=track_data["date"].min(),
        max_date=track_data["date"].max(),
        n_folds=args.n_folds,
        test_horizon_days=horizon_days,
    )
    for f in folds:
        print(f"     {f}")

    # 4. Fit and evaluate models
    print("\n[4/5] Fitting and evaluating models (this is the slow step)...")
    models = [
        HistoricalMean(frequency=cfg["frequency"]),
        SeasonalNaive(season_length=cfg["seasonality"], frequency=cfg["frequency"]),
        DriftNaive(frequency=cfg["frequency"]),
        HoltWintersBaseline(
            season_length=cfg["seasonality"],
            frequency=cfg["frequency"],
            n_jobs=args.n_jobs,
        ),
    ]
    t1 = time.time()
    results = evaluate_models(models, track_data, folds, horizon=horizon_periods)
    print(f"     Evaluation completed in {time.time() - t1:.1f}s")

    # 5. Summarize
    print("\n[5/5] Summary (sorted by WMAPE):")
    summary = aggregate_by_fold(results)
    print(summary)

    # HW-specific coverage report (refit once on full data to get the report)
    print("\n=== Holt-Winters coverage on the last fold ===")
    last_fold = folds[-1]
    train_last = track_data.filter(pl.col("date") <= pl.lit(last_fold.train_end))
    hw_demo = HoltWintersBaseline(
        season_length=cfg["seasonality"],
        frequency=cfg["frequency"],
        n_jobs=args.n_jobs,
    )
    hw_demo.fit(train_last)
    coverage = hw_demo.coverage_report()
    print(
        f"  Total series: {coverage['n_total']:,}\n"
        f"  Handled by AutoETS: {coverage['n_hw']:,} "
        f"({100 * coverage['hw_share']:.1f}%)\n"
        f"  Fallback to DriftNaive: {coverage['n_fallback']:,} "
        f"({100 * (1 - coverage['hw_share']):.1f}%)"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
