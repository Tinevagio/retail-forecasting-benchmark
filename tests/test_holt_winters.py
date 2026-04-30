"""Tests for the Holt-Winters baseline wrapper.

Most tests focus on the routing logic (HW vs fallback) and don't require
statsforecast to actually fit models — that's mocked via sys.modules so
the lazy import inside fit() picks up the mock instead of the real lib.

A single slow integration test verifies end-to-end fitting on a small
synthetic series.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from unittest.mock import MagicMock

import polars as pl
import pytest

from forecasting.models.holt_winters import HoltWintersBaseline


@pytest.fixture
def mock_statsforecast(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a fake `statsforecast` module so the lazy import inside fit()
    picks it up. Returns the StatsForecast mock for assertions.
    """
    mock_sf_class = MagicMock()
    mock_sf_class.return_value.fit = MagicMock()
    mock_sf_class.return_value.predict = MagicMock(return_value=_empty_sf_pandas_df())

    fake_statsforecast = MagicMock()
    fake_statsforecast.StatsForecast = mock_sf_class
    fake_models = MagicMock()
    fake_models.AutoETS = MagicMock()
    fake_statsforecast.models = fake_models

    monkeypatch.setitem(sys.modules, "statsforecast", fake_statsforecast)
    monkeypatch.setitem(sys.modules, "statsforecast.models", fake_models)
    return mock_sf_class


def _empty_sf_pandas_df():
    """Empty DataFrame with the schema statsforecast.predict() returns."""
    import pandas as pd

    return pd.DataFrame({"unique_id": [], "ds": [], "AutoETS": []})


def _make_series(weeks: int, value: float = 10.0, sku_id: str = "X") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [sku_id] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": [value] * weeks,
        }
    )


class TestRoutingLogic:
    """Tests that don't require statsforecast to actually fit anything."""

    def test_short_series_routed_to_fallback(self, mock_statsforecast: MagicMock) -> None:
        """A series shorter than min_history should go to DriftNaive."""
        df = _make_series(weeks=10, sku_id="SHORT")
        model = HoltWintersBaseline(season_length=52, frequency="W", min_history=104)
        model.fit(df)
        # No StatsForecast call because no series qualifies
        mock_statsforecast.assert_not_called()

        coverage = model.coverage_report()
        assert coverage["n_total"] == 1
        assert coverage["n_hw"] == 0
        assert coverage["n_fallback"] == 1

    def test_long_series_routed_to_hw(self, mock_statsforecast: MagicMock) -> None:
        """A series longer than min_history should be routed to AutoETS."""
        df = _make_series(weeks=110, sku_id="LONG")
        model = HoltWintersBaseline(season_length=52, frequency="W", min_history=104)
        model.fit(df)
        # StatsForecast was instantiated (and fit was called on the instance)
        mock_statsforecast.assert_called_once()
        mock_statsforecast.return_value.fit.assert_called_once()

        coverage = model.coverage_report()
        assert coverage["n_hw"] == 1
        assert coverage["n_fallback"] == 0

    def test_mixed_routing(self, mock_statsforecast: MagicMock) -> None:
        """Mix of long and short series should split correctly."""
        long_df = _make_series(weeks=110, sku_id="LONG")
        short_df = _make_series(weeks=10, sku_id="SHORT")
        df = pl.concat([long_df, short_df])

        model = HoltWintersBaseline(season_length=52, frequency="W", min_history=104)
        model.fit(df)

        coverage = model.coverage_report()
        assert coverage["n_total"] == 2
        assert coverage["n_hw"] == 1
        assert coverage["n_fallback"] == 1
        assert coverage["hw_share"] == 0.5

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Call fit"):
            HoltWintersBaseline(season_length=52).predict(horizon=4)

    def test_min_history_default(self) -> None:
        """Default min_history should be 2 * season_length."""
        model = HoltWintersBaseline(season_length=12)
        assert model.min_history == 24


class TestFallbackPredictions:
    """The fallback path goes through DriftNaive — which is well tested already.
    Here we just verify the wiring works.
    """

    def test_fallback_only_predicts_via_drift(self, mock_statsforecast: MagicMock) -> None:
        # Linear trend: 0, 1, 2, ..., 9 -> next 3 weeks should be 10, 11, 12
        df = pl.DataFrame(
            {
                "id": ["X"] * 10,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(10)],
                "sales": [float(i) for i in range(10)],
            }
        )
        # min_history high enough that the series goes to fallback
        model = HoltWintersBaseline(season_length=52, frequency="W", min_history=100)
        model.fit(df)
        preds = model.predict(horizon=3)
        # DriftNaive should produce 10, 11, 12
        assert preds["prediction"].to_list() == [10.0, 11.0, 12.0]

    def test_predict_only_returns_requested_ids(self, mock_statsforecast: MagicMock) -> None:
        df = pl.concat(
            [
                _make_series(weeks=10, value=5.0, sku_id="A"),
                _make_series(weeks=10, value=8.0, sku_id="B"),
            ]
        )
        model = HoltWintersBaseline(season_length=52, frequency="W", min_history=100)
        model.fit(df)
        preds = model.predict(horizon=2, ids=["A"])
        assert set(preds["id"].unique().to_list()) == {"A"}


class TestEdgeCases:
    def test_empty_dataframe_does_not_crash(self, mock_statsforecast: MagicMock) -> None:
        empty = pl.DataFrame(schema={"id": pl.String, "date": pl.Date, "sales": pl.Float64})
        model = HoltWintersBaseline(season_length=52, frequency="W")
        # Empty fit should still work; predict returns empty
        model.fit(empty)
        preds = model.predict(horizon=4)
        assert preds.height == 0

    def test_coverage_on_empty_fit(self, mock_statsforecast: MagicMock) -> None:
        empty = pl.DataFrame(schema={"id": pl.String, "date": pl.Date, "sales": pl.Float64})
        model = HoltWintersBaseline(season_length=52, frequency="W")
        model.fit(empty)
        cov = model.coverage_report()
        assert cov["n_total"] == 0
        # hw_share is NaN for empty fit
        assert cov["hw_share"] != cov["hw_share"]  # NaN is never equal to itself


# ---- Integration test (slow) ---------------------------------------------


@pytest.mark.slow
class TestStatsForecastIntegration:
    """Actually fits AutoETS on a small synthetic series.

    This verifies the full integration with statsforecast and is marked
    `slow` because it takes ~5 seconds. Skip with `pytest -m "not slow"`.
    """

    def test_end_to_end_fit_and_predict(self) -> None:
        # Build a series with clear weekly pattern + trend to make HW happy
        n_weeks = 156  # 3 years
        df = pl.DataFrame(
            {
                "id": ["X"] * n_weeks,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(n_weeks)],
                "sales": [
                    float(50 + i * 0.5 + 10 * (i % 52 == 0))  # trend + yearly spike
                    for i in range(n_weeks)
                ],
            }
        )

        model = HoltWintersBaseline(season_length=52, frequency="W")
        model.fit(df)

        preds = model.predict(horizon=4)
        assert preds.height == 4
        # Predictions must be finite, non-negative
        assert preds["prediction"].is_finite().all()
        assert (preds["prediction"] >= 0).all()
        # Coverage should report the series went through HW (not fallback)
        coverage = model.coverage_report()
        assert coverage["n_hw"] == 1
        assert coverage["n_fallback"] == 0
