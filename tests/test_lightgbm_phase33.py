"""Tests for the LightGBM forecaster — phase 3.3 new features.

These extend the existing test_lightgbm_model.py. Place them in the same
file or in a new test_lightgbm_phase33.py — caller's choice. The existing
tests from phase 3.1 should continue to pass since the changes are
backward-compatible.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.models.lightgbm_model import LightGBMForecaster


def _make_training_df_with_zeros(weeks: int = 60) -> pl.DataFrame:
    """Series with a few suspect zeros mid-history."""
    sales = [float(i % 10 + 5) for i in range(weeks)]
    # Inject 3 stockout-like zeros at indices 30, 35, 40
    for idx in (30, 35, 40):
        sales[idx] = 0.0
    return pl.DataFrame(
        {
            "id": ["X"] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": sales,
            "lag_1": [None, *sales[:-1]],
            "lag_4": [*([None] * 4), *sales[:-4]],
            "month": [(date(2020, 1, 6) + timedelta(days=7 * i)).month for i in range(weeks)],
        }
    )


class TestObjectiveOption:
    def test_tweedie_default_objective(self) -> None:
        """When no objective is specified, tweedie is used (phase 3.3 default)."""
        model = LightGBMForecaster(
            feature_cols=["lag_1"],
            horizon=2,
            n_estimators=10,
        )
        assert model.lgb_params["objective"] == "tweedie"
        assert "tweedie_variance_power" in model.lgb_params

    def test_regression_l1_explicit(self) -> None:
        model = LightGBMForecaster(
            feature_cols=["lag_1"],
            horizon=2,
            n_estimators=10,
            objective="regression_l1",
        )
        assert model.lgb_params["objective"] == "regression_l1"
        # tweedie_variance_power should NOT be present for non-tweedie
        assert "tweedie_variance_power" not in model.lgb_params

    def test_lgb_params_override(self) -> None:
        """User overrides should win over defaults."""
        model = LightGBMForecaster(
            feature_cols=["lag_1"],
            horizon=2,
            n_estimators=10,
            lgb_params={"learning_rate": 0.1, "num_leaves": 31},
        )
        assert model.lgb_params["learning_rate"] == 0.1
        assert model.lgb_params["num_leaves"] == 31


class TestNamingConvention:
    def test_default_name(self) -> None:
        """When using tweedie default with no correction, name reflects it."""
        model = LightGBMForecaster(
            feature_cols=["lag_1"],
            horizon=2,
        )
        assert "tweedie" in model.name.lower()

    def test_name_with_correction(self) -> None:
        model = LightGBMForecaster(
            feature_cols=["lag_1"],
            horizon=2,
            target_correction="rolling_mean",
        )
        assert "rolling_mean" in model.name or "corr" in model.name


class TestTargetCorrection:
    def test_correction_stats_populated_after_fit(self) -> None:
        df = _make_training_df_with_zeros(weeks=60)
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=2,
            n_estimators=20,
            target_correction="rolling_mean",
        )
        model.fit(df)
        assert "n_total_rows" in model.correction_stats
        assert "n_suspicious" in model.correction_stats
        # On this fixture, we expect some zeros to be flagged
        assert model.correction_stats["n_suspicious"] >= 1

    def test_correction_none_stats_zero(self) -> None:
        df = _make_training_df_with_zeros(weeks=60)
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=2,
            n_estimators=10,
            target_correction="none",
        )
        model.fit(df)
        assert model.correction_stats["n_suspicious"] == 0

    def test_predict_works_after_correction(self) -> None:
        df = _make_training_df_with_zeros(weeks=60)
        model = LightGBMForecaster(
            feature_cols=["lag_1", "lag_4", "month"],
            horizon=2,
            n_estimators=20,
            target_correction="rolling_mean",
        )
        model.fit(df)
        preds = model.predict(horizon=2)
        # 1 series x 2 horizons = 2 predictions
        assert preds.height == 2
        assert (preds["prediction"] >= 0).all()
