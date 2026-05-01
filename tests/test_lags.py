"""Tests for lag and rolling-window feature engineering."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.features.lags import (
    add_basic_time_features,
    add_lag_features,
    add_rolling_features,
)


def _make_series(weeks: int = 10, sku_id: str = "X") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [sku_id] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": [float(i + 1) for i in range(weeks)],  # 1.0, 2.0, ..., 10.0
        }
    )


class TestAddLagFeatures:
    def test_lag_1_shifts_one_period(self) -> None:
        df = _make_series(weeks=5)
        result = add_lag_features(df, lags=[1])
        # First row's lag_1 should be null (no previous period)
        assert result["sales_lag_1"][0] is None
        # Subsequent rows should equal the previous sales value
        assert result["sales_lag_1"][1] == 1.0
        assert result["sales_lag_1"][2] == 2.0

    def test_multiple_lags(self) -> None:
        df = _make_series(weeks=5)
        result = add_lag_features(df, lags=[1, 2])
        assert "sales_lag_1" in result.columns
        assert "sales_lag_2" in result.columns
        # Row 3 (sales=3): lag_1 should be 2.0, lag_2 should be 1.0
        assert result["sales_lag_1"][2] == 2.0
        assert result["sales_lag_2"][2] == 1.0

    def test_lag_respects_series_boundaries(self) -> None:
        """A lag from SKU X must not leak into SKU Y."""
        df = pl.concat(
            [
                _make_series(weeks=3, sku_id="X"),
                _make_series(weeks=3, sku_id="Y"),
            ]
        )
        result = add_lag_features(df, lags=[1])
        # First row of each series should have null lag_1
        first_x = result.filter(pl.col("id") == "X").head(1)
        first_y = result.filter(pl.col("id") == "Y").head(1)
        assert first_x["sales_lag_1"][0] is None
        assert first_y["sales_lag_1"][0] is None


class TestAddRollingFeatures:
    def test_rolling_mean_default_shift(self) -> None:
        """With default shift=1, the rolling stat at t uses values strictly
        before t (no leakage)."""
        df = _make_series(weeks=5)  # sales = 1, 2, 3, 4, 5
        result = add_rolling_features(df, windows=[3], statistics=("mean",))
        # At t=3 (sales=4), the 3-period mean of values BEFORE t should be
        # mean(1, 2, 3) = 2.0
        assert result["sales_roll3_mean"][3] == 2.0

    def test_multiple_windows_and_stats(self) -> None:
        df = _make_series(weeks=10)
        result = add_rolling_features(df, windows=[3, 5], statistics=("mean", "max"))
        for col in ["sales_roll3_mean", "sales_roll3_max", "sales_roll5_mean", "sales_roll5_max"]:
            assert col in result.columns

    def test_unknown_statistic_raises(self) -> None:
        df = _make_series()
        with pytest.raises(ValueError, match="Unsupported statistics"):
            add_rolling_features(df, windows=[3], statistics=("mode",))

    def test_no_leakage_first_period(self) -> None:
        """The first period of each series has no past, so rolling stat is null."""
        df = _make_series(weeks=5)
        result = add_rolling_features(df, windows=[3], statistics=("mean",))
        assert result["sales_roll3_mean"][0] is None


class TestBasicTimeFeatures:
    def test_adds_expected_columns(self) -> None:
        df = pl.DataFrame({"date": [date(2020, 3, 15), date(2020, 12, 25)]})
        result = add_basic_time_features(df)
        for col in ["month", "week_of_year", "year", "day_of_year"]:
            assert col in result.columns

    def test_known_values(self) -> None:
        df = pl.DataFrame({"date": [date(2020, 1, 1)]})
        result = add_basic_time_features(df)
        assert result["month"][0] == 1
        assert result["year"][0] == 2020
        assert result["day_of_year"][0] == 1
