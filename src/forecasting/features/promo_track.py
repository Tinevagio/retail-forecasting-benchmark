"""Promo features at weekly granularity.

The existing `forecasting.features.promo` module detects promo weeks via
price drops on the raw weekly price data. Here we build a track-level
joiner that maps the promo signal onto each (id, week) row of an aggregated
track.

Two features are produced:
- `is_on_promo` (Int8): 1 if the (item, store) is on promo this week
- `price_relative_to_ref` (Float64): sell_price / rolling_median_price.
  Default 1.0 when no price data (item not yet referenced).

For sku_state-level tracks, we average the price across stores in the same
state. This is a simplification: in reality, promos can be store-specific.
For sku_store level (the fresh_weekly default), the join is direct.
"""

from __future__ import annotations

import polars as pl

from forecasting.features.promo import detect_promo_from_prices


def add_promo_features_weekly(
    track_df: pl.DataFrame,
    prices: pl.DataFrame,
    calendar: pl.DataFrame,
    id_to_hierarchy: pl.DataFrame,
    track_id_format: str = "item_store",
) -> pl.DataFrame:
    """Add `is_on_promo` and `price_relative_to_ref` to a weekly track.

    Args:
        track_df: Weekly track with id, date (week-start), sales.
        prices: Original M5 sell_prices.csv data.
        calendar: Calendar with d, date, wm_yr_wk.
        id_to_hierarchy: Mapping from track id to (item_id, store_id) or
            (item_id, state_id), as built by `build_id_to_hierarchy`.
        track_id_format: "item_store" or "item_state".

    Returns:
        track_df with added is_on_promo and price_relative_to_ref columns.
    """
    # 1. Detect promos at the (item_id, store_id, wm_yr_wk) level
    promo_signal = detect_promo_from_prices(prices)

    # 2. Map weekly week-start dates to wm_yr_wk via the calendar
    week_start_to_wm = (
        calendar.with_columns(pl.col("date").dt.truncate("1w").alias("week_start"))
        .group_by("week_start")
        .agg(pl.col("wm_yr_wk").first())  # any wm_yr_wk for that week is fine
    )

    # 3. Bring item_id (and store_id or state_id) onto the track
    track_with_hier = track_df.join(id_to_hierarchy, on="id", how="left")

    # 4. Join the wm_yr_wk for each row
    track_with_wm = track_with_hier.join(
        week_start_to_wm, left_on="date", right_on="week_start", how="left"
    )

    if track_id_format == "item_store":
        result = track_with_wm.join(
            promo_signal.select(
                [
                    "item_id",
                    "store_id",
                    "wm_yr_wk",
                    "on_promo",
                    "price_relative_to_ref",
                ]
            ),
            on=["item_id", "store_id", "wm_yr_wk"],
            how="left",
        )
    elif track_id_format == "item_state":
        # For state-level aggregation, average the promo signal across stores
        # in the same state (a coarse but reasonable proxy)
        promo_with_state = promo_signal.join(
            id_to_hierarchy.select(["store_id", "state_id"]).unique(),
            on="store_id",
            how="left",
        )
        promo_state_avg = (
            promo_with_state.group_by(["item_id", "state_id", "wm_yr_wk"])
            .agg(
                [
                    pl.col("on_promo").cast(pl.Float64).mean().alias("on_promo_share"),
                    pl.col("price_relative_to_ref").mean().alias("price_relative_to_ref"),
                ]
            )
            .with_columns((pl.col("on_promo_share") > 0.5).alias("on_promo"))
            .drop("on_promo_share")
        )
        result = track_with_wm.join(
            promo_state_avg.select(
                [
                    "item_id",
                    "state_id",
                    "wm_yr_wk",
                    "on_promo",
                    "price_relative_to_ref",
                ]
            ),
            on=["item_id", "state_id", "wm_yr_wk"],
            how="left",
        )
    else:
        raise ValueError(f"Unknown track_id_format: {track_id_format!r}")

    # 5. Fill nulls (no price data => not on promo, ratio 1.0)
    result = result.with_columns(
        [
            pl.col("on_promo").fill_null(False).cast(pl.Int8).alias("is_on_promo"),
            pl.col("price_relative_to_ref").fill_null(1.0),
        ]
    )

    # 6. Drop the temporary join columns
    drop_cols = ["item_id", "wm_yr_wk", "on_promo"]
    if track_id_format == "item_store":
        drop_cols.extend(["store_id"])
        if "state_id" in result.columns:
            drop_cols.append("state_id")
        if "dept_id" in result.columns:
            drop_cols.append("dept_id")
        if "cat_id" in result.columns:
            drop_cols.append("cat_id")
    elif track_id_format == "item_state":
        drop_cols.extend(["state_id"])
        if "dept_id" in result.columns:
            drop_cols.append("dept_id")
        if "cat_id" in result.columns:
            drop_cols.append("cat_id")

    return result.drop([c for c in drop_cols if c in result.columns])
