"""Promotion-related features derived from sell-price data.

Walmart M5 doesn't expose a promotion flag directly; we infer promo periods
from price drops vs a rolling reference price. This is imperfect but the only
signal available.

Two flavors of features are produced:
- Binary flag `on_promo` (price < ref_price * threshold)
- Continuous `price_relative_to_ref` (= sell_price / ref_price), which is
  generally more informative for ML models than a binary flag.
"""

from __future__ import annotations

import polars as pl


def detect_promo_from_prices(
    prices: pl.DataFrame,
    window_size: int = 8,
    threshold: float = 0.95,
    item_col: str = "item_id",
    store_col: str = "store_id",
    week_col: str = "wm_yr_wk",
    price_col: str = "sell_price",
) -> pl.DataFrame:
    """Flag promo weeks based on price drops vs a rolling reference.

    For each (item, store), computes a rolling median price over the past
    `window_size` weeks and flags weeks where the actual price is below
    `threshold` of that reference.

    Limitations:
    - Misses promos under the threshold (e.g. -3% loyalty discounts)
    - Misses non-price promos (display, placement, bundles)
    - Misidentifies structural price changes as long-running promos until
      the rolling reference catches up

    Args:
        prices: DataFrame with columns item_id, store_id, wm_yr_wk, sell_price.
        window_size: Rolling window length in weeks (default 8).
        threshold: Fraction of reference price below which we flag a promo.
            Default 0.95 = 5% drop.

    Returns:
        Original DataFrame with two added columns:
            - ref_price: rolling-median reference price
            - on_promo: boolean flag
            - price_relative_to_ref: sell_price / ref_price (useful for ML)
    """
    return (
        prices.sort([item_col, store_col, week_col])
        .with_columns(
            pl.col(price_col)
            .rolling_median(window_size=window_size, min_samples=4)
            .over([item_col, store_col])
            .alias("ref_price")
        )
        .with_columns(
            [
                (pl.col(price_col) < threshold * pl.col("ref_price")).alias("on_promo"),
                (pl.col(price_col) / pl.col("ref_price")).alias("price_relative_to_ref"),
            ]
        )
    )


def join_promo_features(
    sales: pl.DataFrame,
    prices_with_promo: pl.DataFrame,
    calendar: pl.DataFrame,
    item_col: str = "item_id",
    store_col: str = "store_id",
    week_col: str = "wm_yr_wk",
    day_col: str = "d",
) -> pl.DataFrame:
    """Join promo features onto a sales DataFrame.

    Sales are daily, prices are weekly. We use the calendar's wm_yr_wk
    column to bridge the two. Days outside the price coverage (e.g. before
    a SKU was introduced in a store) get null promo features, which we
    fill with sensible defaults (no promo, ref price ratio = 1.0).

    Args:
        sales: Long-format daily sales (must contain item_id, store_id, d).
        prices_with_promo: Output of `detect_promo_from_prices()`.
        calendar: Calendar table with d and wm_yr_wk columns.

    Returns:
        Sales DataFrame enriched with on_promo and price_relative_to_ref.
    """
    return (
        sales.join(calendar.select([day_col, week_col]), on=day_col, how="left")
        .join(
            prices_with_promo.select(
                [
                    item_col,
                    store_col,
                    week_col,
                    "on_promo",
                    "price_relative_to_ref",
                ]
            ),
            on=[item_col, store_col, week_col],
            how="left",
        )
        .with_columns(
            [
                pl.col("on_promo").fill_null(False),
                pl.col("price_relative_to_ref").fill_null(1.0),
            ]
        )
    )
