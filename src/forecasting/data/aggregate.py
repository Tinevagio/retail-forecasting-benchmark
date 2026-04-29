"""Temporal and hierarchical aggregation of sales data.

The daily SKU x store granularity is too sparse to forecast directly on most
products (78% have >50% zeros, see EDA). Real-world replenishment decisions
are made at coarser aggregations:

- Fresh products: weekly per SKU x store (or SKU x warehouse)
- Dry / non-food: monthly per SKU x state (or SKU x warehouse)

This module provides building blocks to aggregate either temporally
(daily -> weekly/monthly) or hierarchically (store -> state -> national)
or both.

A "track" in this project corresponds to a (frequency, hierarchy_level,
category_filter) triplet. The function `prepare_track()` packages all this
into one call.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

Frequency = Literal["W", "M"]
HierarchyLevel = Literal["sku_store", "sku_state", "sku_national", "dept_state"]


def aggregate_temporal(
    df: pl.DataFrame,
    frequency: Frequency,
    date_col: str = "date",
    sales_col: str = "sales",
    group_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Sum daily sales into weekly or monthly buckets.

    Args:
        df: Long-format daily sales.
        frequency: "W" for weekly (week starts Monday), "M" for monthly.
        date_col: Name of the date column.
        sales_col: Name of the sales column to sum.
        group_cols: Columns to keep alongside the time aggregation (e.g.
            ["id", "item_id", "store_id"]). If None, only date is kept.

    Returns:
        DataFrame with one row per (group, period). The date column is
        replaced by the start-of-period date.
    """
    if group_cols is None:
        group_cols = []

    # Polars truncate gives us the start of week (Monday) or start of month
    truncate_unit = "1w" if frequency == "W" else "1mo"

    return (
        df.with_columns(pl.col(date_col).dt.truncate(truncate_unit).alias(date_col))
        .group_by([*group_cols, date_col])
        .agg(pl.col(sales_col).sum())
        .sort([*group_cols, date_col])
    )


def aggregate_hierarchical(
    df: pl.DataFrame,
    level: HierarchyLevel,
    sales_col: str = "sales",
    date_col: str = "date",
) -> pl.DataFrame:
    """Aggregate daily sales to a coarser hierarchical level.

    The input must contain the columns: id, item_id, dept_id, cat_id,
    store_id, state_id, date, sales. Output schema depends on `level`:

    - sku_store: identity (no aggregation), key = (item_id, store_id)
    - sku_state: sum across stores in the same state, key = (item_id, state_id)
    - sku_national: sum across all stores, key = item_id
    - dept_state: sum within (dept, state), key = (dept_id, state_id)

    Args:
        df: Long-format daily sales with the standard M5 columns.
        level: Aggregation target.

    Returns:
        DataFrame with the appropriate key columns + date + summed sales.
        The series identifier `id` is rebuilt from the key for downstream use.
    """
    keys: dict[HierarchyLevel, list[str]] = {
        "sku_store": ["item_id", "store_id"],
        "sku_state": ["item_id", "state_id"],
        "sku_national": ["item_id"],
        "dept_state": ["dept_id", "state_id"],
    }
    key_cols = keys[level]

    aggregated = (
        df.group_by([*key_cols, date_col]).agg(pl.col(sales_col).sum()).sort([*key_cols, date_col])
    )

    # Rebuild a synthetic `id` column so downstream code can treat it uniformly
    id_expr = pl.concat_str(key_cols, separator="__").alias("id")
    return aggregated.with_columns(id_expr)


# Track configuration: maps friendly names to aggregation parameters.
# Aligned with the project README and EDA findings.
TRACK_CONFIGS: dict[str, dict] = {
    "fresh_weekly": {
        "frequency": "W",
        "level": "sku_store",
        "categories": ["FOODS_3"],
        "horizon_periods": 4,  # 4 weeks
        "seasonality": 52,
    },
    "dry_monthly": {
        "frequency": "M",
        "level": "sku_state",
        "categories": ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"],
        "horizon_periods": 3,  # 3 months
        "seasonality": 12,
    },
    "non_food_monthly": {
        "frequency": "M",
        "level": "sku_state",
        "categories": ["HOBBIES_1", "HOBBIES_2"],
        "horizon_periods": 3,
        "seasonality": 12,
    },
}


def prepare_track(
    df: pl.DataFrame,
    track_name: str,
    cat_col: str = "dept_id",
    date_col: str = "date",
    sales_col: str = "sales",
) -> pl.DataFrame:
    """One-shot pipeline: filter to the track's categories, aggregate
    hierarchically, then aggregate temporally.

    Args:
        df: Long-format daily sales (item_id, dept_id, cat_id, store_id,
            state_id, date, sales as minimum).
        track_name: One of TRACK_CONFIGS keys.
        cat_col: Column used for category filtering. Default "dept_id"
            because the EDA's track mapping uses dept-level codes (FOODS_3, etc.).

    Returns:
        DataFrame with columns id (synthetic), date (period start), sales.
    """
    if track_name not in TRACK_CONFIGS:
        raise KeyError(f"Unknown track {track_name!r}. Known tracks: {list(TRACK_CONFIGS)}")

    cfg = TRACK_CONFIGS[track_name]

    # 1. Filter
    filtered = df.filter(pl.col(cat_col).is_in(cfg["categories"]))
    if filtered.height == 0:
        raise ValueError(
            f"No rows after filtering on {cat_col} in {cfg['categories']}. "
            f"Check that the column exists and the filter values match."
        )

    # 2. Hierarchical aggregation
    hier = aggregate_hierarchical(filtered, cfg["level"], sales_col, date_col)

    # 3. Temporal aggregation
    final = aggregate_temporal(hier, cfg["frequency"], date_col, sales_col, group_cols=["id"])
    return final
