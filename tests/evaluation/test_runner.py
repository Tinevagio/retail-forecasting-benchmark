"""Tests for forecasting.evaluation.runner with return_predictions option."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Self

import polars as pl

from forecasting.data.splits import make_walk_forward_folds
from forecasting.evaluation.runner import (
    PREDICTIONS_COLUMNS,
    aggregate_by_fold,
    evaluate_model,
    evaluate_models,
)
from forecasting.models.base import Forecaster


class _ConstantForecaster(Forecaster):
    """Toy model that always predicts a constant. Used for runner tests."""

    def __init__(self, value: float = 1.0, name: str = "const") -> None:
        self.value = value
        self._name = name
        self._ids: list[str] = []
        self._last_date: date | None = None

    def fit(self, train_df: pl.DataFrame) -> Self:
        self._ids = train_df["id"].unique().sort().to_list()
        self._last_date = train_df["date"].max()  # type: ignore[assignment]
        return self

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        target_ids = ids if ids is not None else self._ids
        if self._last_date is None:
            raise RuntimeError("Predict before fit")
        # Daily horizon: predict for last_date + 1 .. last_date + horizon
        rows: list[dict[str, object]] = []
        for sid in target_ids:
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        "id": sid,
                        "date": self._last_date + timedelta(days=h),
                        "prediction": float(self.value),
                    }
                )
        return pl.DataFrame(rows)

    @property
    def name(self) -> str:
        return self._name


def _build_panel(n_ids: int = 5, n_days: int = 60) -> pl.DataFrame:
    start = date(2016, 1, 1)
    rows = []
    for i in range(n_ids):
        for d in range(n_days):
            rows.append(
                {
                    "id": f"ID_{i:03d}",
                    "date": start + timedelta(days=d),
                    "sales": (i + d) % 10,
                }
            )
    return pl.DataFrame(rows)


class TestEvaluateModelDefault:
    def test_default_returns_metrics_only(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        result = evaluate_model(_ConstantForecaster(), df, folds, horizon=7)
        # Single dataframe (not a tuple)
        assert isinstance(result, pl.DataFrame)
        assert "wmape" in result.columns
        assert "fold_id" in result.columns
        assert result.height == 2  # 2 folds

    def test_default_signature_unchanged(self) -> None:
        """Existing call sites pass no kw — must still work."""
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        # Positional call as in the existing scripts
        evaluate_model(_ConstantForecaster(), df, folds, 7)


class TestEvaluateModelWithPredictions:
    def test_returns_tuple(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        result = evaluate_model(
            _ConstantForecaster(), df, folds, horizon=7, return_predictions=True
        )
        assert isinstance(result, tuple)
        metrics, predictions = result
        assert isinstance(metrics, pl.DataFrame)
        assert isinstance(predictions, pl.DataFrame)

    def test_predictions_schema(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        _, predictions = evaluate_model(
            _ConstantForecaster(), df, folds, horizon=7, return_predictions=True
        )
        assert tuple(predictions.columns) == PREDICTIONS_COLUMNS
        # 3 ids x 7 days x 2 folds = 42 rows
        assert predictions.height == 42

    def test_predictions_carry_y_true_and_y_pred(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=1,
            test_horizon_days=7,
        )
        _, predictions = evaluate_model(
            _ConstantForecaster(value=2.5),
            df,
            folds,
            horizon=7,
            return_predictions=True,
        )
        assert (predictions["y_pred"] == 2.5).all()
        # y_true comes from the test slice and should match the panel rule
        assert predictions["y_true"].dtype.is_integer()

    def test_metrics_match_between_modes(self) -> None:
        """The metrics returned must be identical with and without
        return_predictions=True."""
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        metrics_only = evaluate_model(_ConstantForecaster(), df, folds, horizon=7)
        metrics_with_pred, _ = evaluate_model(
            _ConstantForecaster(), df, folds, horizon=7, return_predictions=True
        )
        assert metrics_only.to_dicts() == metrics_with_pred.to_dicts()


class TestEvaluateModelsMultiple:
    def test_default_concatenates_metrics(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        models = [
            _ConstantForecaster(value=1.0, name="m1"),
            _ConstantForecaster(value=2.0, name="m2"),
        ]
        result = evaluate_models(models, df, folds, horizon=7)
        assert isinstance(result, pl.DataFrame)
        # 2 models x 2 folds
        assert result.height == 4
        assert set(result["model_name"].unique().to_list()) == {"m1", "m2"}

    def test_with_predictions_concatenates_both(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        models = [
            _ConstantForecaster(value=1.0, name="m1"),
            _ConstantForecaster(value=2.0, name="m2"),
        ]
        metrics, predictions = evaluate_models(
            models, df, folds, horizon=7, return_predictions=True
        )
        assert metrics.height == 4
        # 2 models x 3 ids x 7 days x 2 folds = 84 rows
        assert predictions.height == 84
        assert set(predictions["model_name"].unique().to_list()) == {"m1", "m2"}

    def test_aggregate_by_fold_still_works(self) -> None:
        df = _build_panel(n_ids=3, n_days=60)
        folds = make_walk_forward_folds(
            min_date=df["date"].min(),  # type: ignore[arg-type]
            max_date=df["date"].max(),  # type: ignore[arg-type]
            n_folds=2,
            test_horizon_days=7,
        )
        result = evaluate_models([_ConstantForecaster()], df, folds, horizon=7)
        summary = aggregate_by_fold(result)
        assert summary.height == 1
        assert "wmape_mean" in summary.columns
