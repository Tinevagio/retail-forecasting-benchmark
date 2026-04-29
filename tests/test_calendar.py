"""Tests for calendar feature engineering."""

from __future__ import annotations

import polars as pl

from forecasting.features.calendar import (
    add_basic_calendar_features,
    add_event_features,
    add_snap_features,
)


class TestBasicCalendarFeatures:
    def test_adds_expected_columns(self) -> None:
        df = pl.DataFrame({"date": ["2024-01-15", "2024-06-30"]}).with_columns(
            pl.col("date").str.to_date()
        )
        result = add_basic_calendar_features(df)
        for col in ["dow", "month", "year", "day_of_month", "is_weekend"]:
            assert col in result.columns

    def test_known_values(self) -> None:
        # 2024-01-15 is a Monday (dow=1 in Polars, Mon=1..Sun=7)
        # 2024-06-30 is a Sunday (dow=7)
        df = pl.DataFrame({"date": ["2024-01-15", "2024-06-30"]}).with_columns(
            pl.col("date").str.to_date()
        )
        result = add_basic_calendar_features(df)
        assert result["dow"][0] == 1  # Monday
        assert result["dow"][1] == 7  # Sunday
        assert result["is_weekend"][0] is False
        assert result["is_weekend"][1] is True
        assert result["month"][0] == 1
        assert result["year"][0] == 2024


class TestSnapFeatures:
    def test_picks_state_specific_snap(self) -> None:
        sales = pl.DataFrame(
            {
                "id": ["X_CA", "X_TX", "X_WI"],
                "state_id": ["CA", "TX", "WI"],
                "d": ["d_1", "d_1", "d_1"],
                "sales": [1, 1, 1],
            }
        )
        # On d_1: CA has SNAP, TX does not, WI has SNAP
        calendar = pl.DataFrame(
            {
                "d": ["d_1"],
                "snap_CA": [1],
                "snap_TX": [0],
                "snap_WI": [1],
            }
        )
        result = add_snap_features(sales, calendar)
        # Match on state -> the right SNAP column should drive the result
        ca_row = result.filter(pl.col("state_id") == "CA")
        tx_row = result.filter(pl.col("state_id") == "TX")
        wi_row = result.filter(pl.col("state_id") == "WI")
        assert ca_row["is_snap_day"][0] is True
        assert tx_row["is_snap_day"][0] is False
        assert wi_row["is_snap_day"][0] is True

    def test_drops_intermediate_columns(self) -> None:
        sales = pl.DataFrame({"id": ["X"], "state_id": ["CA"], "d": ["d_1"], "sales": [1]})
        calendar = pl.DataFrame(
            {
                "d": ["d_1"],
                "snap_CA": [1],
                "snap_TX": [0],
                "snap_WI": [0],
            }
        )
        result = add_snap_features(sales, calendar)
        assert "snap_CA" not in result.columns
        assert "snap_TX" not in result.columns
        assert "snap_WI" not in result.columns


class TestEventFeatures:
    def test_joins_event_columns(self) -> None:
        sales = pl.DataFrame({"id": ["X", "Y"], "d": ["d_1", "d_2"], "sales": [1, 1]})
        calendar = pl.DataFrame(
            {
                "d": ["d_1", "d_2"],
                "event_name_1": ["SuperBowl", None],
                "event_type_1": ["Sporting", None],
            }
        )
        result = add_event_features(sales, calendar)
        assert "event_name" in result.columns
        assert "event_type" in result.columns
        # Event row
        event_row = result.filter(pl.col("d") == "d_1")
        assert event_row["event_name"][0] == "SuperBowl"
        assert event_row["event_type"][0] == "Sporting"
        # Non-event row
        non_event = result.filter(pl.col("d") == "d_2")
        assert non_event["event_name"][0] is None
