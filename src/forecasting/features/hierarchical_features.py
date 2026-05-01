"""Hierarchical features: information transfer across siblings.

The EDA showed that 77% of series are cold-start, with limited individual
history. Hierarchical aggregates let a model "see" what's happening at the
department or store level, even when the individual SKU has weak history.
This is one of the most powerful levers for ML over Holt-Winters, which
is strictly univariate.

All features use **lagged values only** to avoid leakage: at time t, we
expose the department/store/category mean from time t-1 (or earlier).

Implementation note:
The track's `id` is a synthetic concatenation (e.g. "ITEM__STORE"). To
reconstruct the hierarchy, we expect the caller to provide a separate
mapping DataFrame with the original M5 columns (item_id, dept_id, cat_id,
store_id, state_id).
"""

from __future__ import annotations

import polars as pl


def add_hierarchical_lag_features(
    track_df: pl.DataFrame,
    id_to_hierarchy: pl.DataFrame,
    lag: int = 1,
    levels: tuple[str, ...] = ("dept_id", "store_id", "cat_id"),
) -> pl.DataFrame:
    """Add lagged mean-sales features at each hierarchy level.

    For each level (dept, store, cat), computes the average sales of all
    series in that level at period t-lag, and joins it onto each row.

    Args:
        track_df: Weekly track with columns id, date, sales.
        id_to_hierarchy: Mapping from id to hierarchy columns. Must contain
            "id" plus all columns listed in `levels`. Built once by the caller.
        lag: How many periods back to look. Default 1 (previous period).
        levels: Hierarchy columns to aggregate over.

    Returns:
        track_df with added columns `{level}_avg_sales_lag_{lag}` for each
        level. Null when the lag falls before the dataset's start.
    """
    # Validate
    missing = set(levels) - set(id_to_hierarchy.columns)
    if missing:
        raise ValueError(f"id_to_hierarchy missing columns: {missing}")

    # Attach hierarchy info to the track once
    track_h = track_df.join(
        id_to_hierarchy.select(["id", *levels]),
        on="id",
        how="left",
    )

    enriched = track_h
    for level in levels:
        # Compute mean sales per (level, date)
        level_means = (
            track_h.group_by([level, "date"])
            .agg(pl.col("sales").mean().alias(f"{level}_avg_sales"))
            .sort([level, "date"])
            .with_columns(
                pl.col(f"{level}_avg_sales")
                .shift(lag)
                .over(level)
                .alias(f"{level}_avg_sales_lag_{lag}")
            )
            .drop(f"{level}_avg_sales")
        )

        enriched = enriched.join(
            level_means,
            on=[level, "date"],
            how="left",
        )

    # Drop the joined hierarchy columns; we only wanted them as join keys
    return enriched.drop(list(levels))


def build_id_to_hierarchy(
    daily_sales: pl.DataFrame,
    track_id_format: str = "item_store",
) -> pl.DataFrame:
    """Build the (id -> hierarchy columns) mapping for a given track.

    The track's `id` is built by `aggregate_hierarchical()` as a concatenation
    of key columns. This helper reconstructs the mapping back to the original
    M5 hierarchy columns so that hierarchical features can be computed.

    Args:
        daily_sales: Original long-format daily sales (must have item_id,
            dept_id, cat_id, store_id, state_id).
        track_id_format: How the track's id was built. Currently supported:
            - "item_store": id = item_id + "__" + store_id (sku_store level)
            - "item_state": id = item_id + "__" + state_id (sku_state level)

    Returns:
        DataFrame with columns: id, item_id, dept_id, cat_id, store_id, state_id.
    """
    if track_id_format == "item_store":
        return daily_sales.select(
            [
                pl.concat_str(["item_id", "store_id"], separator="__").alias("id"),
                "item_id",
                "dept_id",
                "cat_id",
                "store_id",
                "state_id",
            ]
        ).unique()
    if track_id_format == "item_state":
        return daily_sales.select(
            [
                pl.concat_str(["item_id", "state_id"], separator="__").alias("id"),
                "item_id",
                "dept_id",
                "cat_id",
                "state_id",
            ]
        ).unique()
    raise ValueError(f"Unknown track_id_format: {track_id_format!r}")
