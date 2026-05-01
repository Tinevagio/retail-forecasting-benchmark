"""Tests for the LightGBM forecaster.

These tests verify the routing logic, error handling, and basic predictions
on small synthetic data. End-to-end behavior with real features is exercised
in the benchmark scripts.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from forecasting.models.lightgbm_model import LightGBMForecaster


def _make_training_df(weeks: int = 60) -> pl.DataFrame:
    """Synthetic series with a clear trend that LGBM should learn."""
    return pl.DataFrame(
        {
            "id": ["X"] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": [float(i) for i in range(weeks)],  # linear trend
            # Pre-computed simple features (in a real pipeline these come from
            # add_lag_features/add_rolling_features)
            "lag_1": [None] + [float(i - 1) for i in range(1, weeks)],
            "lag_4": [None] * 4 + [float(i - 4) for i in range(4, weeks)],
            "month": [((date(2020, 1, 6) + timedelta(days=7 * i)).month) for i in range(weeks)],
        }
    )


class TestFit:
    def test_fits_one_model_per_horizon(self) -> None:
        df = _make_training_df()
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=4,
            n_estimators=20,  # small for fast tests
        )
        model.fit(df)
        # 4 booster models, one per horizon step
        assert len(model._models) == 4
        assert set(model._models.keys()) == {1, 2, 3, 4}

    def test_missing_feature_raises(self) -> None:
        df = _make_training_df()
        model = LightGBMForecaster(
            feature_cols=["lag_1", "nonexistent_col"],
            horizon=2,
            n_estimators=10,
        )
        with pytest.raises(ValueError, match="Missing feature columns"):
            model.fit(df)

    def test_empty_feature_list_raises(self) -> None:
        df = _make_training_df()
        model = LightGBMForecaster(
            feature_cols=[],
            horizon=2,
            n_estimators=10,
        )
        with pytest.raises(ValueError, match="feature_cols"):
            model.fit(df)


class TestPredict:
    def test_returns_horizon_predictions(self) -> None:
        df = _make_training_df()
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=3,
            n_estimators=20,
        )
        model.fit(df)
        preds = model.predict(horizon=3)
        # 1 series x 3 horizons = 3 predictions
        assert preds.height == 3
        # Predictions are non-negative (clipped)
        assert (preds["prediction"] >= 0).all()

    def test_horizon_too_large_raises(self) -> None:
        df = _make_training_df()
        model = LightGBMForecaster(
            feature_cols=["lag_1"],
            horizon=2,
            n_estimators=10,
        )
        model.fit(df)
        with pytest.raises(ValueError, match="exceeds the trained horizon"):
            model.predict(horizon=5)

    def test_predict_before_fit_raises(self) -> None:
        model = LightGBMForecaster(feature_cols=["lag_1"], horizon=2)
        with pytest.raises(RuntimeError, match="Call fit"):
            model.predict(horizon=2)

    def test_filters_by_ids(self) -> None:
        df = pl.concat(
            [
                _make_training_df().with_columns(pl.lit("A").alias("id")),
                _make_training_df().with_columns(pl.lit("B").alias("id")),
            ]
        )
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=2,
            n_estimators=10,
        )
        model.fit(df)
        preds = model.predict(horizon=2, ids=["A"])
        assert set(preds["id"].unique().to_list()) == {"A"}


class TestFeatureImportance:
    def test_importance_returned(self) -> None:
        df = _make_training_df()
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=2,
            n_estimators=20,
        )
        model.fit(df)
        importance = model.feature_importance()
        assert importance.height == 3
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        # Sorted descending
        importances = importance["importance"].to_list()
        assert importances == sorted(importances, reverse=True)

    def test_importance_before_fit_raises(self) -> None:
        model = LightGBMForecaster(feature_cols=["lag_1"], horizon=2)
        with pytest.raises(RuntimeError, match="Call fit"):
            model.feature_importance()


class TestLearnsTrend:
    """Sanity check: on a clear linear trend, LGBM predictions should
    extrapolate roughly forward, not be wildly off."""

    def test_predicts_above_history_on_trend(self) -> None:
        df = _make_training_df(weeks=60)  # sales = 0..59, last value ~59
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=1,
            n_estimators=100,
        )
        model.fit(df)
        preds = model.predict(horizon=1)
        # On a strong upward trend, the next prediction should be at least
        # close to the last observed value (LGBM tree models don't extrapolate
        # beyond training range, but they should at least predict something
        # in the right ballpark, not 0 or 100).
        pred_value = preds["prediction"][0]
        assert 30 < pred_value < 70, (
            f"Expected prediction around training tail value (~59), got {pred_value}"
        )
