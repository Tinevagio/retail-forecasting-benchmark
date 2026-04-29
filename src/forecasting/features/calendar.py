"""Calendar-derived features for daily forecasting.

Holt-Winters captures basic seasonality (DOW, monthly) implicitly. For ML
models, we expose these explicitly along with calendar events and SNAP days,
which are the primary blind spots of HW.

Design choice: events are encoded **per name** rather than aggregated by
type. The EDA showed that aggregating events masks heterogeneous effects
(closures vs boosters), so per-event encoding is essential.
"""

from __future__ import annotations

import polars as pl


def add_basic_calendar_features(
    df: pl.DataFrame,
    date_col: str = "date",
) -> pl.DataFrame:
    """Add day-of-week, month, year, and is-weekend features.

    Args:
        df: DataFrame with a date column.
        date_col: Name of the date column.

    Returns:
        DataFrame with added columns: dow, month, year, day_of_month, is_weekend.
    """
    return df.with_columns(
        [
            pl.col(date_col).dt.weekday().alias("dow"),
            pl.col(date_col).dt.month().alias("month"),
            pl.col(date_col).dt.year().alias("year"),
            pl.col(date_col).dt.day().alias("day_of_month"),
            (pl.col(date_col).dt.weekday() >= 6).alias("is_weekend"),
        ]
    )


def add_snap_features(
    df: pl.DataFrame,
    calendar: pl.DataFrame,
    state_col: str = "state_id",
    day_col: str = "d",
) -> pl.DataFrame:
    """Add a unified `is_snap_day` column based on the SKU's state.

    The M5 calendar has separate snap_CA, snap_TX, snap_WI columns. This helper
    picks the right column based on each row's state_id, producing a single
    `is_snap_day` boolean.

    Args:
        df: Sales DataFrame with at least state_id and d columns.
        calendar: Calendar table with snap_CA, snap_TX, snap_WI, d columns.

    Returns:
        DataFrame with added column is_snap_day (bool).
    """
    enriched = df.join(
        calendar.select([day_col, "snap_CA", "snap_TX", "snap_WI"]),
        on=day_col,
        how="left",
    )
    return enriched.with_columns(
        pl.when(pl.col(state_col) == "CA")
        .then(pl.col("snap_CA") == 1)
        .when(pl.col(state_col) == "TX")
        .then(pl.col("snap_TX") == 1)
        .when(pl.col(state_col) == "WI")
        .then(pl.col("snap_WI") == 1)
        .otherwise(False)
        .alias("is_snap_day")
    ).drop(["snap_CA", "snap_TX", "snap_WI"])


def add_event_features(
    df: pl.DataFrame,
    calendar: pl.DataFrame,
    day_col: str = "d",
) -> pl.DataFrame:
    """Add per-named-event features.

    Brings event_name_1 and event_type_1 from the calendar onto the sales
    DataFrame. The name allows for per-event encoding downstream (recommended
    based on EDA findings); the type provides a fallback for rare events.

    Args:
        df: Sales DataFrame with a d column.
        calendar: Calendar table with event_name_1 and event_type_1 columns.

    Returns:
        DataFrame with added columns: event_name and event_type.
    """
    return df.join(
        calendar.select(
            [
                day_col,
                pl.col("event_name_1").alias("event_name"),
                pl.col("event_type_1").alias("event_type"),
            ]
        ),
        on=day_col,
        how="left",
    )
