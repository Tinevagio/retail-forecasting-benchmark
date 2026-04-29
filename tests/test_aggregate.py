"""Tests for temporal and hierarchical aggregation."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.data.aggregate import (
    TRACK_CONFIGS,
    aggregate_hierarchical,
    aggregate_temporal,
    prepare_track,
)


def _make_daily_df() -> pl.DataFrame:
    """Two SKUs in two stores over 30 days, sales = 1 every day."""
    rows = []
    for item in ["A", "B"]:
        for store in ["S1", "S2"]:
            for d in range(30):
                rows.append(
                    {
                        "item_id": item,
                        "dept_id": "FOODS_3",
                        "cat_id": "FOODS",
                        "store_id": store,
                        "state_id": "CA",
                        "date": date(2020, 1, 1) + timedelta(days=d),
                        "sales": 1,
                    }
                )
    return pl.DataFrame(rows)


class TestAggregateTemporal:
    def test_weekly_sum(self) -> None:
        df = _make_daily_df()
        weekly = aggregate_temporal(df, frequency="W", group_cols=["item_id", "store_id"])
        # Each (item, store) had 30 days of 1 -> ~5 weeks of 7 days
        # Most weeks will have 7 days summed = 7
        full_weeks = weekly.filter(pl.col("sales") == 7)
        assert full_weeks.height >= 4  # at least 4 full weeks per group

    def test_monthly_sum(self) -> None:
        df = _make_daily_df()
        monthly = aggregate_temporal(df, frequency="M", group_cols=["item_id", "store_id"])
        # All 30 days fall in January 2020 -> single row per group, sum=30
        assert monthly.height == 4  # 2 items x 2 stores
        assert (monthly["sales"] == 30).all()


class TestAggregateHierarchical:
    def test_sku_state_aggregation(self) -> None:
        df = _make_daily_df()
        agg = aggregate_hierarchical(df, level="sku_state")
        # 2 items x 1 state x 30 days = 60 rows; daily sales of 2 (sum across 2 stores)
        assert agg.height == 60
        assert (agg["sales"] == 2).all()
        assert "id" in agg.columns
        # id format: item__state
        assert set(agg["id"].unique()) == {"A__CA", "B__CA"}

    def test_sku_national_aggregation(self) -> None:
        df = _make_daily_df()
        agg = aggregate_hierarchical(df, level="sku_national")
        # 2 items x 30 days, daily sales = 2 stores x 1 = 2
        assert agg.height == 60
        assert (agg["sales"] == 2).all()
        assert set(agg["id"].unique()) == {"A", "B"}


class TestPrepareTrack:
    def test_known_track_runs(self) -> None:
        df = _make_daily_df()
        # The default 'fresh_weekly' filters dept_id == FOODS_3
        out = prepare_track(df, "fresh_weekly")
        assert out.height > 0
        assert "id" in out.columns
        assert "date" in out.columns
        assert "sales" in out.columns

    def test_unknown_track_raises(self) -> None:
        df = _make_daily_df()
        with pytest.raises(KeyError, match="Unknown track"):
            prepare_track(df, "nonexistent_track")

    def test_filter_with_no_match_raises(self) -> None:
        # df has dept_id='FOODS_3', track 'non_food_monthly' filters HOBBIES_*
        df = _make_daily_df()
        with pytest.raises(ValueError, match="No rows after filtering"):
            prepare_track(df, "non_food_monthly")


class TestTrackConfigs:
    def test_all_tracks_have_required_keys(self) -> None:
        required = {"frequency", "level", "categories", "horizon_periods", "seasonality"}
        for name, cfg in TRACK_CONFIGS.items():
            missing = required - set(cfg)
            assert not missing, f"Track {name} missing keys: {missing}"
