"""Aggregate calendar events and SNAP days to weekly granularity.

The fresh_weekly track sums daily sales into weeks. Calendar features must
follow the same aggregation: instead of a daily `is_snap_day` flag, we want
per-week features like "how many SNAP days in this week" and "which events
fall in this week".

The EDA identified specific events as significant:
- High-impact closures: Christmas, Thanksgiving (negative effect)
- Genuine demand boosters: Pesach End, Purim End, Labor Day
- Sporting: Super Bowl (positive on FOODS)

Other events are aggregated into a single `event_other_in_week` flag to
keep the feature set manageable.
"""

from __future__ import annotations

import polars as pl

# Events identified in EDA as having clear individual signal
NAMED_EVENTS_OF_INTEREST: tuple[str, ...] = (
    "Christmas",  # closure, sales drop to ~0
    "Thanksgiving",  # partial closure, sales ~40% of normal
    "PesachEnd",  # demand booster (top of leaderboard in EDA)
    "PurimEnd",  # demand booster
    "LaborDay",  # demand booster
    "SuperBowl",  # sporting boost on FOODS
)


def add_weekly_snap_features(
    track_df: pl.DataFrame,
    calendar: pl.DataFrame,
    daily_sales_with_state: pl.DataFrame,
) -> pl.DataFrame:
    """Add `snap_days_in_week` based on the SKU's state.

    Process:
    1. Per (state, week), count how many days are SNAP days
    2. For each track row, look up its state from the daily sales mapping
    3. Join the appropriate count

    Args:
        track_df: The aggregated weekly track data (id, date, sales).
        calendar: Calendar table with d, date, snap_CA, snap_TX, snap_WI.
        daily_sales_with_state: Original long-format daily sales with the
            id and state_id columns, used to map id -> state_id.

    Returns:
        track_df with added column `snap_days_in_week` (Int).
    """
    # Build (id -> state) mapping. Each id is unique to one state in M5.
    id_to_state = daily_sales_with_state.select(["id", "state_id"]).unique()

    # The track's id format depends on the hierarchical level used.
    # For sku_store level, id = item_id__store_id; we recover the store
    # via a join on the original (item_id, store_id) -> state_id mapping.
    # We delegate that complexity by accepting a precomputed mapping that
    # the caller produces once.

    # Per-week SNAP day counts per state
    cal_with_week = calendar.with_columns(pl.col("date").dt.truncate("1w").alias("week_start"))
    snap_per_week_state = cal_with_week.group_by("week_start").agg(
        [
            pl.col("snap_CA").sum().alias("snap_days_CA"),
            pl.col("snap_TX").sum().alias("snap_days_TX"),
            pl.col("snap_WI").sum().alias("snap_days_WI"),
        ]
    )

    # Join state info onto track via the id mapping
    track_with_state = track_df.join(id_to_state, on="id", how="left")

    # Join the per-week SNAP counts (week_start = track's date column)
    enriched = track_with_state.join(
        snap_per_week_state,
        left_on="date",
        right_on="week_start",
        how="left",
    )

    # Pick the right SNAP count based on state
    enriched = enriched.with_columns(
        pl.when(pl.col("state_id") == "CA")
        .then(pl.col("snap_days_CA"))
        .when(pl.col("state_id") == "TX")
        .then(pl.col("snap_days_TX"))
        .when(pl.col("state_id") == "WI")
        .then(pl.col("snap_days_WI"))
        .otherwise(0)
        .fill_null(0)
        .cast(pl.Int64)
        .alias("snap_days_in_week")
    )

    # Drop intermediate columns
    return enriched.drop(["state_id", "snap_days_CA", "snap_days_TX", "snap_days_WI"])


def add_weekly_event_features(
    track_df: pl.DataFrame,
    calendar: pl.DataFrame,
    named_events: tuple[str, ...] = NAMED_EVENTS_OF_INTEREST,
) -> pl.DataFrame:
    """Add per-named-event flags + an `event_other_in_week` catch-all.

    The flags are per-week: `event_<NAME>_in_week` is 1 if any day in the
    week has that named event, 0 otherwise. The track's date column is
    expected to be the week-start (Monday) date.

    Args:
        track_df: Weekly track data with a date column at week-start.
        calendar: Calendar with d, date, event_name_1, event_type_1.
        named_events: Tuple of event names to encode individually. Other
            events are pooled into `event_other_in_week`.

    Returns:
        track_df with added columns:
            - event_<NAME>_in_week (Int8) for each name
            - event_other_in_week (Int8) for any other event
            - any_event_in_week (Int8) for any event (closure-detection helper)
    """
    cal_events = calendar.filter(pl.col("event_name_1").is_not_null()).select(
        [
            pl.col("date").dt.truncate("1w").alias("week_start"),
            pl.col("event_name_1"),
        ]
    )

    # For each (week, named_event), 1 if present
    event_flags = []
    for ev in named_events:
        flag_col = (
            cal_events.filter(pl.col("event_name_1") == ev)
            .group_by("week_start")
            .agg(pl.lit(1).alias(f"event_{ev}_in_week"))
        )
        event_flags.append(flag_col)

    # Other events catch-all
    other_flag = (
        cal_events.filter(~pl.col("event_name_1").is_in(list(named_events)))
        .group_by("week_start")
        .agg(pl.lit(1).alias("event_other_in_week"))
    )

    # Any event flag (regardless of name) - useful as closure proxy
    any_flag = cal_events.group_by("week_start").agg(pl.lit(1).alias("any_event_in_week"))

    # Successive left joins onto track_df, filling nulls with 0
    enriched = track_df
    for ef in event_flags:
        enriched = enriched.join(ef, left_on="date", right_on="week_start", how="left")
    enriched = enriched.join(other_flag, left_on="date", right_on="week_start", how="left")
    enriched = enriched.join(any_flag, left_on="date", right_on="week_start", how="left")

    # Fill the joined nulls and cast to Int8 for memory efficiency
    flag_cols = [f"event_{ev}_in_week" for ev in named_events] + [
        "event_other_in_week",
        "any_event_in_week",
    ]
    return enriched.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in flag_cols])
