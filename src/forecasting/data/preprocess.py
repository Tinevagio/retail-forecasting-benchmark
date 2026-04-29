"""Preprocessing utilities: stockout detection, history truncation, etc.

These functions transform raw sales data into model-ready signals,
correcting for known supply-chain artifacts (stockouts, store closures, ...).
"""

from __future__ import annotations

import polars as pl


def detect_suspicious_zeros(
    df: pl.DataFrame,
    window_size: int = 15,
    threshold: float = 1.0,
    id_col: str = "id",
    sales_col: str = "sales",
    date_col: str = "date",
) -> pl.DataFrame:
    """Flag zero-sale days that are likely stockouts rather than absent demand.

    Heuristic: a zero is "suspicious" when surrounding sales (centered rolling
    mean over `window_size` days) are above `threshold`. The intuition is that
    a true zero-demand period would show zero sales nearby too; isolated zeros
    among normal sales activity are more consistent with stockouts.

    Note: this heuristic is conservative — it cannot detect long contiguous
    stockout periods (rolling mean drops to zero too) nor very low-volume
    products (rolling mean rarely exceeds threshold). For production use,
    combine with a price-availability check (no price = SKU not referenced).

    Args:
        df: Long-format sales DataFrame with at least id, date, sales columns.
        window_size: Centered rolling window in days (default 15).
        threshold: Minimum rolling mean to flag the zero (default 1.0).
        id_col: Name of the series identifier column.
        sales_col: Name of the sales column.
        date_col: Name of the date column (must be sortable).

    Returns:
        Same DataFrame with two additional columns:
            - rolling_mean: centered rolling mean of sales
            - suspicious_zero: boolean flag for likely-stockout days
    """
    return (
        df.sort([id_col, date_col])
        .with_columns(
            [
                pl.col(sales_col)
                .rolling_mean(window_size=window_size, center=True, min_samples=10)
                .over(id_col)
                .alias("rolling_mean"),
            ]
        )
        .with_columns(
            ((pl.col(sales_col) == 0) & (pl.col("rolling_mean") > threshold)).alias(
                "suspicious_zero"
            )
        )
    )


def stockout_summary_by_sku(
    flagged: pl.DataFrame,
    id_col: str = "id",
    sales_col: str = "sales",
) -> pl.DataFrame:
    """Aggregate stockout statistics per SKU x store series.

    Args:
        flagged: Output of `detect_suspicious_zeros()`.
        id_col: Name of the series identifier column.
        sales_col: Name of the sales column.

    Returns:
        DataFrame with one row per series and columns:
            - n_suspicious: number of suspicious zero days
            - n_zeros: total number of zero-sale days
            - avg_sales: mean sales (excluding nulls)
            - suspicious_share: fraction of zeros that are suspicious
    """
    return (
        flagged.group_by(id_col)
        .agg(
            [
                pl.col("suspicious_zero").sum().alias("n_suspicious"),
                (pl.col(sales_col) == 0).sum().alias("n_zeros"),
                pl.col(sales_col).mean().alias("avg_sales"),
            ]
        )
        .with_columns(
            pl.when(pl.col("n_zeros") > 0)
            .then(pl.col("n_suspicious") / pl.col("n_zeros").cast(pl.Float64))
            .otherwise(0.0)
            .alias("suspicious_share")
        )
        .sort("n_suspicious", descending=True)
    )
