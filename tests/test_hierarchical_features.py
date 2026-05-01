"""Tests for hierarchical lag features."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from forecasting.features.hierarchical_features import (
    add_hierarchical_lag_features,
    build_id_to_hierarchy,
)


def _make_track() -> pl.DataFrame:
    """4 SKUs in 2 departments x 2 stores, 3 weeks each."""
    rows = []
    week_dates = [date(2020, 1, 6), date(2020, 1, 13), date(2020, 1, 20)]
    skus = ["A__S1", "A__S2", "B__S1", "B__S2"]
    for sku in skus:
        for d in week_dates:
            rows.append({"id": sku, "date": d, "sales": 10.0})
    return pl.DataFrame(rows)


def _make_hierarchy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["A__S1", "A__S2", "B__S1", "B__S2"],
            "item_id": ["A", "A", "B", "B"],
            "store_id": ["S1", "S2", "S1", "S2"],
            "dept_id": ["FOODS_3", "FOODS_3", "FOODS_3", "FOODS_3"],
            "cat_id": ["FOODS", "FOODS", "FOODS", "FOODS"],
            "state_id": ["CA", "CA", "CA", "CA"],
        }
    )


class TestAddHierarchicalLagFeatures:
    def test_adds_expected_columns(self) -> None:
        track = _make_track()
        hier = _make_hierarchy()
        result = add_hierarchical_lag_features(track, hier, lag=1, levels=("dept_id", "store_id"))
        assert "dept_id_avg_sales_lag_1" in result.columns
        assert "store_id_avg_sales_lag_1" in result.columns

    def test_first_period_is_null(self) -> None:
        """The first period has no lag-1 history -> nulls expected."""
        track = _make_track()
        hier = _make_hierarchy()
        result = add_hierarchical_lag_features(track, hier, lag=1, levels=("dept_id",))
        first_period_rows = result.filter(pl.col("date") == date(2020, 1, 6))
        assert first_period_rows["dept_id_avg_sales_lag_1"].null_count() == first_period_rows.height

    def test_lag_value_is_correct(self) -> None:
        """All 4 SKUs have sales=10 in week 1, so dept mean at lag 1 should be 10
        when we're looking at week 2."""
        track = _make_track()
        hier = _make_hierarchy()
        result = add_hierarchical_lag_features(track, hier, lag=1, levels=("dept_id",))
        second_week_rows = result.filter(pl.col("date") == date(2020, 1, 13))
        # All rows in week 2 should see lag_1 = 10 for the dept
        assert (second_week_rows["dept_id_avg_sales_lag_1"] == 10.0).all()

    def test_missing_hierarchy_column_raises(self) -> None:
        track = _make_track()
        hier = _make_hierarchy()
        with pytest.raises(ValueError, match="missing columns"):
            add_hierarchical_lag_features(track, hier, lag=1, levels=("nonexistent_col",))


class TestBuildIdToHierarchy:
    def test_item_store_format(self) -> None:
        daily = pl.DataFrame(
            {
                "item_id": ["A", "A", "B"],
                "store_id": ["S1", "S2", "S1"],
                "dept_id": ["D1", "D1", "D2"],
                "cat_id": ["C1", "C1", "C2"],
                "state_id": ["CA", "CA", "TX"],
            }
        )
        result = build_id_to_hierarchy(daily, "item_store")
        assert result.height == 3
        assert "id" in result.columns
        ids = sorted(result["id"].to_list())
        assert ids == ["A__S1", "A__S2", "B__S1"]

    def test_unknown_format_raises(self) -> None:
        daily = pl.DataFrame(
            {
                "item_id": ["A"],
                "store_id": ["S1"],
                "dept_id": ["D1"],
                "cat_id": ["C1"],
                "state_id": ["CA"],
            }
        )
        with pytest.raises(ValueError, match="Unknown track_id_format"):
            build_id_to_hierarchy(daily, "unknown_format")
