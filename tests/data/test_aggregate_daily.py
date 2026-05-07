"""Tests for forecasting.data.aggregate (modifications for daily + sampling)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.data.aggregate import (
    TRACK_CONFIGS,
    aggregate_temporal,
    prepare_track,
)


def _build_daily_m5_like(
    n_items: int = 10,
    n_stores: int = 3,
    n_days: int = 30,
    cats: list[str] | None = None,
    states: list[str] | None = None,
) -> pl.DataFrame:
    """Build a synthetic dataframe in the M5 long format."""
    if cats is None:
        cats = ["FOODS", "HOUSEHOLD", "HOBBIES"]
    if states is None:
        states = ["CA", "TX", "WI"]

    start = date(2016, 1, 1)
    rows = []
    for i in range(n_items):
        cat = cats[i % len(cats)]
        dept = f"{cat}_1"  # one dept per cat for simplicity
        for s in range(n_stores):
            state = states[s % len(states)]
            for d in range(n_days):
                rows.append(
                    {
                        "item_id": f"ITEM_{i:04d}",
                        "dept_id": dept,
                        "cat_id": cat,
                        "store_id": f"{state}_{s + 1}",
                        "state_id": state,
                        "date": start + timedelta(days=d),
                        "sales": (i + s + d) % 5,
                    }
                )
    return pl.DataFrame(rows)


class TestAggregateTemporalDaily:
    def test_daily_is_noop_on_clean_data(self) -> None:
        df = pl.DataFrame(
            {
                "id": ["A", "A", "B", "B"],
                "date": [
                    date(2016, 1, 1),
                    date(2016, 1, 2),
                    date(2016, 1, 1),
                    date(2016, 1, 2),
                ],
                "sales": [3, 5, 7, 9],
            }
        )
        result = aggregate_temporal(df, frequency="D", group_cols=["id"])
        assert result.height == 4
        # Same dates as input (no truncation)
        assert sorted(result["date"].to_list()) == sorted(df["date"].to_list())
        # Same sales (no aggregation)
        assert sorted(result["sales"].to_list()) == sorted(df["sales"].to_list())

    def test_daily_dedupes_duplicates(self) -> None:
        """If the input has duplicate (id, date) rows, they should be summed."""
        df = pl.DataFrame(
            {
                "id": ["A", "A"],
                "date": [date(2016, 1, 1), date(2016, 1, 1)],
                "sales": [3, 5],
            }
        )
        result = aggregate_temporal(df, frequency="D", group_cols=["id"])
        assert result.height == 1
        assert result["sales"][0] == 8

    def test_weekly_still_truncates(self) -> None:
        """Sanity: existing W behaviour is preserved."""
        df = pl.DataFrame(
            {
                "id": ["A", "A", "A"],
                "date": [date(2016, 1, 4), date(2016, 1, 5), date(2016, 1, 11)],
                "sales": [3, 5, 7],
            }
        )
        result = aggregate_temporal(df, frequency="W", group_cols=["id"])
        # Week of Jan 4 (Mon) sums to 8, week of Jan 11 to 7
        assert result.height == 2
        sums = sorted(result["sales"].to_list())
        assert sums == [7, 8]


class TestDailySampleTrack:
    def test_config_is_present(self) -> None:
        assert "daily_sample" in TRACK_CONFIGS
        cfg = TRACK_CONFIGS["daily_sample"]
        assert cfg["frequency"] == "D"
        assert cfg["level"] == "sku_store"
        assert "sample" in cfg
        assert cfg["sample"]["n_skus"] == 1000
        assert cfg["sample"]["seed"] == 42

    def test_prepare_track_applies_sampling(self) -> None:
        # Build a synthetic population larger than the sample target
        df = _build_daily_m5_like(n_items=100, n_stores=3, n_days=10)
        # Override sample to a tractable size for the test
        original = TRACK_CONFIGS["daily_sample"]["sample"]
        TRACK_CONFIGS["daily_sample"]["sample"] = {
            "n_skus": 30,
            "strata": ["cat_id", "state_id"],
            "seed": 42,
        }
        try:
            result = prepare_track(df, "daily_sample")
            # 30 unique series x 10 days = 300 rows
            assert result["id"].n_unique() == 30
            assert result.height == 30 * 10
            # Schema: id, date, sales
            assert set(result.columns) == {"id", "date", "sales"}
            # Date range preserved
            assert result["date"].min() == date(2016, 1, 1)
            assert result["date"].max() == date(2016, 1, 10)
        finally:
            TRACK_CONFIGS["daily_sample"]["sample"] = original

    def test_prepare_track_sampling_is_deterministic(self) -> None:
        df = _build_daily_m5_like(n_items=100, n_stores=3, n_days=5)
        original = TRACK_CONFIGS["daily_sample"]["sample"]
        TRACK_CONFIGS["daily_sample"]["sample"] = {
            "n_skus": 30,
            "strata": ["cat_id", "state_id"],
            "seed": 42,
        }
        try:
            r1 = prepare_track(df, "daily_sample")
            r2 = prepare_track(df, "daily_sample")
            assert r1["id"].unique().sort().to_list() == r2["id"].unique().sort().to_list()
        finally:
            TRACK_CONFIGS["daily_sample"]["sample"] = original

    def test_existing_tracks_unaffected(self) -> None:
        """Backward-compat sanity: fresh_weekly still works."""
        df = _build_daily_m5_like(n_items=10, n_stores=3, n_days=21, cats=["FOODS"])
        # Patch the dept_id to FOODS_3 so the FOODS_3 filter passes
        df = df.with_columns(pl.lit("FOODS_3").alias("dept_id"))
        result = prepare_track(df, "fresh_weekly")
        assert result.height > 0
        assert set(result.columns) == {"id", "date", "sales"}


class TestUnknownTrack:
    def test_unknown_track_raises(self) -> None:
        df = _build_daily_m5_like(n_items=2)
        with pytest.raises(KeyError, match="Unknown track"):
            prepare_track(df, "not_a_track")
