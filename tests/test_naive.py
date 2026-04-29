"""Tests for naive baseline forecasters."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.models.naive import DriftNaive, HistoricalMean, SeasonalNaive


def _make_weekly_series() -> pl.DataFrame:
    """One series with 20 weekly observations starting 2020-01-06 (Monday)."""
    return pl.DataFrame(
        {
            "id": ["X"] * 20,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(20)],
            "sales": [10.0] * 20,
        }
    )


class TestHistoricalMean:
    def test_predicts_mean(self) -> None:
        df = _make_weekly_series()
        model = HistoricalMean(frequency="W").fit(df)
        preds = model.predict(horizon=4)
        assert preds.height == 4
        assert (preds["prediction"] == 10.0).all()

    def test_filters_by_ids(self) -> None:
        df = pl.DataFrame(
            {
                "id": ["X"] * 5 + ["Y"] * 5,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(5)] * 2,
                "sales": [10.0] * 5 + [20.0] * 5,
            }
        )
        model = HistoricalMean(frequency="W").fit(df)
        preds = model.predict(horizon=2, ids=["Y"])
        assert preds.height == 2
        assert (preds["id"] == "Y").all()
        assert (preds["prediction"] == 20.0).all()

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Call fit"):
            HistoricalMean().predict(horizon=4)


class TestSeasonalNaive:
    def test_predicts_lookback_value(self) -> None:
        # Build 60 weeks where sales = week_index, so lookback at -52 weeks
        # lets us know exactly what to expect
        df = pl.DataFrame(
            {
                "id": ["X"] * 60,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(60)],
                "sales": list(range(60)),
            }
        )
        model = SeasonalNaive(season_length=52, frequency="W").fit(df)
        preds = model.predict(horizon=4)
        # The next 4 weeks (indices 60..63) should map back to weeks 8..11
        # which have sales 8, 9, 10, 11
        assert preds["prediction"].to_list() == [8, 9, 10, 11]

    def test_falls_back_to_mean_when_history_too_short(self) -> None:
        df = _make_weekly_series()  # only 20 weeks of history
        model = SeasonalNaive(season_length=52, frequency="W").fit(df)
        preds = model.predict(horizon=4)
        # Insufficient history -> fallback to mean (which is 10.0)
        assert preds.height == 4
        assert (preds["prediction"] == 10.0).all()

    def test_invalid_season_length_raises(self) -> None:
        with pytest.raises(ValueError):
            SeasonalNaive(season_length=0)


class TestDriftNaive:
    def test_extrapolates_linear_trend(self) -> None:
        # Sales: 0, 1, 2, ..., 9 over 10 weeks. slope = 1, last = 9
        # Next 3 weeks should be 10, 11, 12
        df = pl.DataFrame(
            {
                "id": ["X"] * 10,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(10)],
                "sales": list(range(10)),
            }
        )
        model = DriftNaive(frequency="W").fit(df)
        preds = model.predict(horizon=3)
        assert preds["prediction"].to_list() == [10, 11, 12]

    def test_clips_negative_predictions(self) -> None:
        # Decreasing series ending at 1 with slope -2: would predict -1, -3, ...
        # We clip to 0 since negative sales are nonsensical.
        df = pl.DataFrame(
            {
                "id": ["X"] * 5,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(5)],
                "sales": [9.0, 7.0, 5.0, 3.0, 1.0],
            }
        )
        model = DriftNaive(frequency="W").fit(df)
        preds = model.predict(horizon=3)
        # Should never go below 0
        assert (preds["prediction"] >= 0).all()
