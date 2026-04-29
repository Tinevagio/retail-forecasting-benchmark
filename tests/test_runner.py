"""Tests for the cross-validation evaluation runner."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.data.splits import make_walk_forward_folds
from forecasting.evaluation.runner import (
    aggregate_by_fold,
    evaluate_model,
    evaluate_models,
)
from forecasting.models.naive import HistoricalMean, SeasonalNaive


def _make_series(weeks: int = 30, value: float = 10.0) -> pl.DataFrame:
    """Single series, weekly, constant sales."""
    return pl.DataFrame(
        {
            "id": ["X"] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": [value] * weeks,
        }
    )


class TestEvaluateModel:
    def test_perfect_constant_series(self) -> None:
        """On constant sales, HistoricalMean should achieve WMAPE of 0."""
        df = _make_series()
        # Folds: 4 folds of 14 days = 2 weeks each. We forecast 2 weeks per fold.
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),
            max_date=df["date"].max(),
            n_folds=4,
            test_horizon_days=14,
        )
        model = HistoricalMean(frequency="W")
        results = evaluate_model(model, df, folds, horizon=2)
        assert results.height == 4
        # WMAPE should be 0 (or very close) for a constant series
        assert (results["wmape"] < 1e-9).all()

    def test_returns_expected_columns(self) -> None:
        df = _make_series()
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),
            max_date=df["date"].max(),
            n_folds=2,
            test_horizon_days=14,
        )
        results = evaluate_model(HistoricalMean(frequency="W"), df, folds, horizon=2)
        expected_cols = {
            "model_name",
            "fold_id",
            "train_end",
            "test_start",
            "test_end",
            "wmape",
            "bias",
            "rmse",
            "mae",
            "n_obs",
        }
        assert expected_cols <= set(results.columns)


class TestEvaluateModels:
    def test_concatenates_multiple_models(self) -> None:
        df = _make_series(weeks=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),
            max_date=df["date"].max(),
            n_folds=3,
            test_horizon_days=14,
        )
        models = [
            HistoricalMean(frequency="W"),
            SeasonalNaive(season_length=4, frequency="W"),
        ]
        results = evaluate_models(models, df, folds, horizon=2)
        # 2 models x 3 folds = 6 rows
        assert results.height == 6
        assert set(results["model_name"].unique()) == {"HistoricalMean", "SeasonalNaive"}


class TestAggregateByFold:
    def test_summary_one_row_per_model(self) -> None:
        df = _make_series(weeks=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),
            max_date=df["date"].max(),
            n_folds=3,
            test_horizon_days=14,
        )
        models = [HistoricalMean(frequency="W"), SeasonalNaive(season_length=4, frequency="W")]
        results = evaluate_models(models, df, folds, horizon=2)
        summary = aggregate_by_fold(results)
        assert summary.height == 2
        assert "wmape_mean" in summary.columns
        assert "wmape_std" in summary.columns
        # Sorted by wmape_mean ascending
        wmapes = summary["wmape_mean"].to_list()
        assert wmapes == sorted(wmapes)
