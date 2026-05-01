"""Lag and rolling-window features for time-series ML.

These are the most fundamental features for any forecasting model. The
lag features answer "what happened k periods ago", the rolling features
answer "what's the recent trend/level".

Design choices:
- Features are computed per-series (group by id) to respect series boundaries
- All operations are leakage-safe by construction: only past values are used
- Polars `over(...)` does the per-series partitioning natively, much faster
  than pandas groupby+apply
- Rolling windows are right-aligned (default in Polars), so the value at
  time t is computed from values strictly before t
"""

from __future__ import annotations

import polars as pl


def add_lag_features(
    df: pl.DataFrame,
    lags: list[int],
    target_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
) -> pl.DataFrame:
    """Add lag features: value of `target_col` at t-k for each k in `lags`.

    Args:
        df: Long-format DataFrame, sorted by (id, date) or sortable.
        lags: List of lag periods. E.g. [1, 4, 52] = 1 week, 4 weeks, 52 weeks ago.
        target_col: Column to lag.
        id_col: Series identifier column.
        date_col: Date column (used for sorting).

    Returns:
        DataFrame with new columns named `{target_col}_lag_{k}` for each k.
    """
    df_sorted = df.sort([id_col, date_col])
    return df_sorted.with_columns(
        [pl.col(target_col).shift(k).over(id_col).alias(f"{target_col}_lag_{k}") for k in lags]
    )


def add_rolling_features(
    df: pl.DataFrame,
    windows: list[int],
    target_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
    statistics: tuple[str, ...] = ("mean", "std"),
    shift: int = 1,
) -> pl.DataFrame:
    """Add rolling-window aggregation features.

    Args:
        df: Long-format DataFrame.
        windows: List of window sizes. E.g. [4, 12, 52] = 4-week, 12-week, 52-week
            rolling stats.
        target_col: Column to aggregate over.
        id_col: Series identifier.
        date_col: Date column for sorting.
        statistics: Which stats to compute. Supported: mean, std, min, max, median.
        shift: How many periods to shift the rolling window AFTER computing,
            to avoid using the current period's value. Default 1 (rolling stats
            from period t use values up to t-1).

    Returns:
        DataFrame with new columns `{target_col}_roll{w}_{stat}` for each
        combination of window and statistic.
    """
    stat_to_expr = {
        "mean": lambda c, w: c.rolling_mean(window_size=w, min_samples=1),
        "std": lambda c, w: c.rolling_std(window_size=w, min_samples=2),
        "min": lambda c, w: c.rolling_min(window_size=w, min_samples=1),
        "max": lambda c, w: c.rolling_max(window_size=w, min_samples=1),
        "median": lambda c, w: c.rolling_median(window_size=w, min_samples=1),
    }
    unknown = [s for s in statistics if s not in stat_to_expr]
    if unknown:
        raise ValueError(f"Unsupported statistics: {unknown}")

    df_sorted = df.sort([id_col, date_col])
    new_cols = []
    for w in windows:
        for stat in statistics:
            expr = stat_to_expr[stat](pl.col(target_col), w).over(id_col)
            if shift > 0:
                expr = expr.shift(shift).over(id_col)
            new_cols.append(expr.alias(f"{target_col}_roll{w}_{stat}"))
    return df_sorted.with_columns(new_cols)


def add_basic_time_features(
    df: pl.DataFrame,
    date_col: str = "date",
) -> pl.DataFrame:
    """Add simple calendar features extractable from the date alone.

    Phase 3.2 will add more sophisticated calendar features (per-event
    encoding, SNAP, etc.). This is the bare minimum for a first ML run.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with added: month, week_of_year, year, day_of_year.
    """
    return df.with_columns(
        [
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.week().alias("week_of_year"),
            pl.col(date_col).dt.year().alias("year"),
            pl.col(date_col).dt.ordinal_day().alias("day_of_year"),
        ]
    )
