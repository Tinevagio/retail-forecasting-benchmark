"""Tests for weekly SNAP and event feature aggregation."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.features.event_features import (
    NAMED_EVENTS_OF_INTEREST,
    add_weekly_event_features,
    add_weekly_snap_features,
)


def _make_calendar(n_days: int = 21) -> pl.DataFrame:
    """A 3-week calendar starting Mon 2020-01-06."""
    return pl.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, n_days + 1)],
            "date": [date(2020, 1, 6) + timedelta(days=i) for i in range(n_days)],
            "snap_CA": [1 if i < 5 else 0 for i in range(n_days)],
            "snap_TX": [0] * n_days,
            "snap_WI": [0] * n_days,
            "event_name_1": [None] * n_days,
            "event_type_1": [None] * n_days,
        }
    )


def _make_weekly_track() -> pl.DataFrame:
    """Two SKUs, 3 weekly observations each."""
    week_starts = [date(2020, 1, 6), date(2020, 1, 13), date(2020, 1, 20)]
    return pl.DataFrame(
        {
            "id": ["A__S1"] * 3 + ["B__S1"] * 3,
            "date": week_starts * 2,
            "sales": [10.0, 12.0, 11.0, 5.0, 6.0, 7.0],
        }
    )


def _make_id_state_mapping() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["A__S1", "B__S1"],
            "state_id": ["CA", "TX"],
        }
    )


class TestAddWeeklySnapFeatures:
    def test_counts_snap_days_per_week_for_correct_state(self) -> None:
        track = _make_weekly_track()
        cal = _make_calendar(n_days=21)
        # A__S1 is in CA -> first 5 days are SNAP -> first week has 5 SNAP days
        # B__S1 is in TX -> never SNAP -> 0 SNAP days

        # Build the daily df with state info that the function expects
        daily = pl.DataFrame(
            {
                "id": ["A__S1", "B__S1"],
                "state_id": ["CA", "TX"],
            }
        )

        result = add_weekly_snap_features(track, cal, daily)
        a_first_week = result.filter(
            (pl.col("id") == "A__S1") & (pl.col("date") == date(2020, 1, 6))
        )
        b_first_week = result.filter(
            (pl.col("id") == "B__S1") & (pl.col("date") == date(2020, 1, 6))
        )
        assert a_first_week["snap_days_in_week"][0] == 5
        assert b_first_week["snap_days_in_week"][0] == 0

    def test_returns_zero_for_unknown_state(self) -> None:
        track = _make_weekly_track()
        cal = _make_calendar()
        # SKU with a state not in CA/TX/WI
        daily = pl.DataFrame(
            {
                "id": ["A__S1", "B__S1"],
                "state_id": ["XX", "YY"],
            }
        )
        result = add_weekly_snap_features(track, cal, daily)
        assert (result["snap_days_in_week"] == 0).all()


class TestAddWeeklyEventFeatures:
    def test_adds_expected_columns(self) -> None:
        track = _make_weekly_track()
        cal = _make_calendar()
        result = add_weekly_event_features(track, cal)
        for ev in NAMED_EVENTS_OF_INTEREST:
            assert f"event_{ev}_in_week" in result.columns
        assert "event_other_in_week" in result.columns
        assert "any_event_in_week" in result.columns

    def test_no_events_in_calendar_yields_all_zeros(self) -> None:
        track = _make_weekly_track()
        cal = _make_calendar()  # no events
        result = add_weekly_event_features(track, cal)
        assert (result["any_event_in_week"] == 0).all()
        assert (result["event_other_in_week"] == 0).all()

    def test_named_event_flagged_in_correct_week(self) -> None:
        track = _make_weekly_track()
        cal = _make_calendar()
        # Add Christmas on day 5 (which is Friday Jan 10, week 1)
        cal = cal.with_columns(
            pl.when(pl.col("d") == "d_5")
            .then(pl.lit("Christmas"))
            .otherwise(pl.col("event_name_1"))
            .alias("event_name_1")
        )
        result = add_weekly_event_features(track, cal)
        first_week = result.filter(pl.col("date") == date(2020, 1, 6))
        assert first_week["event_Christmas_in_week"][0] == 1
        assert first_week["any_event_in_week"][0] == 1

    def test_unknown_event_falls_into_other(self) -> None:
        track = _make_weekly_track()
        cal = _make_calendar()
        cal = cal.with_columns(
            pl.when(pl.col("d") == "d_5")
            .then(pl.lit("UnknownEvent"))
            .otherwise(pl.col("event_name_1"))
            .alias("event_name_1")
        )
        result = add_weekly_event_features(track, cal)
        first_week = result.filter(pl.col("date") == date(2020, 1, 6))
        assert first_week["event_other_in_week"][0] == 1
        assert first_week["any_event_in_week"][0] == 1
        # Named events should still be 0
        for ev in NAMED_EVENTS_OF_INTEREST:
            assert first_week[f"event_{ev}_in_week"][0] == 0
