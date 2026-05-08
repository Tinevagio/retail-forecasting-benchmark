"""Build the `daily_sample` track parquet for downstream stock optimization.

Pipeline:
    1. Load the M5 calendar and sales (long format).
    2. Apply prepare_track("daily_sample") which:
       - filters to the configured categories
       - applies stratified sampling on (cat_id, state_id) -> 1000 series
       - aggregates hierarchically (sku_store identity, here)
       - aggregates temporally (no-op for D)
    3. Write to data/processed/track_daily_sample.parquet.

Why this exists
---------------
The benchmark scripts (run_holt_winters.py, run_lightgbm_phase33.py) accept a
`--track` flag and rely on TRACK_CONFIGS for the pipeline. Materializing the
track as a parquet lets the project 2 consumer (retail-stock-optimization)
download a single artifact rather than re-running this pipeline.

Usage
-----
    uv run python scripts/build_track_daily_sample.py
    uv run python scripts/build_track_daily_sample.py --output custom_path.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from forecasting.data.aggregate import TRACK_CONFIGS, prepare_track
from forecasting.data.load import load_calendar, load_sales, melt_sales

DEFAULT_OUTPUT = Path("data/processed/track_daily_sample.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to write the track parquet (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--print-summary",
        action="store_true",
        help="Print sample composition by strata after building.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = TRACK_CONFIGS["daily_sample"]
    sample_cfg = cfg.get("sample")
    if sample_cfg is None:
        # Defensive: should not happen with the current TRACK_CONFIGS
        print("ERROR: daily_sample track has no sample spec.", file=sys.stderr)
        return 1

    print(
        f"Building daily_sample track (n_skus={sample_cfg['n_skus']}, seed={sample_cfg['seed']})..."
    )

    t0 = time.time()
    print("[1/3] Loading calendar and sales...")
    calendar = load_calendar()
    sales_wide = load_sales()
    sales = melt_sales(sales_wide, calendar=calendar)
    print(f"      Long-format sales: {sales.shape}, loaded in {time.time() - t0:.1f}s")

    print("[2/3] Applying prepare_track('daily_sample')...")
    t1 = time.time()
    track = prepare_track(sales, "daily_sample")
    print(f"      Track built in {time.time() - t1:.1f}s")
    print(f"      Series: {track['id'].n_unique():,}, rows: {track.height:,}")
    print(f"      Date range: {track['date'].min()} -> {track['date'].max()}")

    print(f"[3/3] Writing to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    track.write_parquet(args.output)
    size_mb = args.output.stat().st_size / 1e6
    print(f"      Wrote {size_mb:.1f} MB.")

    if args.print_summary:
        # Strata composition: map back to the original strata via item_id/store_id
        # parsed from the composite id.
        import polars as pl

        composite = track.select("id").unique().sort("id")
        # Reconstruct the strata by joining on the original sales df
        item_store = composite.with_columns(
            pl.col("id")
            .str.split_exact("__", 1)
            .struct.rename_fields(["item_id", "store_id"])
            .alias("parts")
        ).unnest("parts")
        strata = (
            sales.select("item_id", "store_id", "cat_id", "state_id")
            .unique()
            .join(item_store, on=["item_id", "store_id"], how="inner")
        )
        composition = (
            strata.group_by("cat_id", "state_id")
            .agg(pl.len().alias("n_series"))
            .sort("cat_id", "state_id")
        )
        print("\nSample composition by strata:")
        print(composition)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
