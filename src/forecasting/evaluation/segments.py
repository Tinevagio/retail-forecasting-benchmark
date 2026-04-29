"""Segmentation utilities for forecast evaluation.

Aggregate metrics hide a lot. The EDA showed dramatic heterogeneity across
intermittence levels, categories, and promo status. Phase 4 will report
metrics by segment, not just headlines. These helpers produce the segment
labels.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

IntermittenceLabel = Literal[
    "high_volume",  # < 10% zeros
    "medium",  # 10-50% zeros
    "intermittent",  # 50-80% zeros
    "extreme_intermittent",  # > 80% zeros
]


def classify_intermittence(zero_share: float) -> IntermittenceLabel:
    """Map a zero-fraction value to a discrete intermittence category.

    Thresholds are derived from EDA observations on M5: the natural
    breakpoints are at 10% (very rare for a series), 50% (more zeros than
    sales), and 80% (extreme tail).

    Args:
        zero_share: Fraction of zero-sale days, in [0, 1].

    Returns:
        Categorical label.

    Raises:
        ValueError: if zero_share is outside [0, 1].
    """
    if not 0.0 <= zero_share <= 1.0:
        raise ValueError(f"zero_share must be in [0, 1], got {zero_share}")
    if zero_share < 0.10:
        return "high_volume"
    if zero_share < 0.50:
        return "medium"
    if zero_share < 0.80:
        return "intermittent"
    return "extreme_intermittent"


def add_intermittence_labels(
    df: pl.DataFrame,
    id_col: str = "id",
    sales_col: str = "sales",
) -> pl.DataFrame:
    """Compute zero-share and intermittence label per series, joined back.

    Args:
        df: Long-format sales DataFrame.
        id_col: Name of the series identifier.
        sales_col: Name of the sales column.

    Returns:
        DataFrame with two added columns: zero_share (Float64) and
        intermittence_label (Categorical).
    """
    per_series = df.group_by(id_col).agg((pl.col(sales_col) == 0).mean().alias("zero_share"))

    # Polars doesn't easily support row-wise Python function application,
    # so we replicate the bucket logic with when/then expressions.
    per_series = per_series.with_columns(
        pl.when(pl.col("zero_share") < 0.10)
        .then(pl.lit("high_volume"))
        .when(pl.col("zero_share") < 0.50)
        .then(pl.lit("medium"))
        .when(pl.col("zero_share") < 0.80)
        .then(pl.lit("intermittent"))
        .otherwise(pl.lit("extreme_intermittent"))
        .alias("intermittence_label")
    )

    return df.join(per_series, on=id_col, how="left")


def compute_lift(
    df: pl.DataFrame,
    group_col: str,
    condition_col: str,
    value_col: str = "sales",
) -> pl.DataFrame:
    """Compute the % lift on `value_col` when `condition_col` is True vs False.

    Useful for quantifying the effect of a binary feature (on_promo, is_event,
    is_snap_day) at different aggregation levels.

    Args:
        df: DataFrame containing all columns referenced.
        group_col: Column to group by (e.g. cat_id).
        condition_col: Boolean column whose True/False slices we compare.
        value_col: Numeric column to average within each slice.

    Returns:
        DataFrame with columns: group_col, false_avg, true_avg, lift_pct.
        Sorted by lift descending.
    """
    pivoted = (
        df.group_by([group_col, condition_col])
        .agg(pl.col(value_col).mean().alias("avg_value"))
        .pivot(values="avg_value", index=group_col, on=condition_col)
    )

    # The pivot column names may be "true"/"false" strings or boolean.
    # Normalize to expected names.
    rename_map = {}
    for c in pivoted.columns:
        if c == group_col:
            continue
        c_lower = str(c).lower()
        if c_lower in ("true", "1"):
            rename_map[c] = "true_avg"
        elif c_lower in ("false", "0"):
            rename_map[c] = "false_avg"
    pivoted = pivoted.rename(rename_map)

    return pivoted.with_columns(
        ((pl.col("true_avg") / pl.col("false_avg") - 1) * 100).alias("lift_pct")
    ).sort("lift_pct", descending=True)
