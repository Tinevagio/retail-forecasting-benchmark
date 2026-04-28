"""Tests for forecasting metrics."""

import math

import numpy as np

from forecasting.evaluation.metrics import (
    bias,
    compute_all_metrics,
    mae,
    rmse,
    wmape,
)


class TestWMAPE:
    def test_perfect_prediction(self) -> None:
        y_true = [10, 20, 30]
        y_pred = [10, 20, 30]
        assert wmape(y_true, y_pred) == 0.0

    def test_known_value(self) -> None:
        # |10-12| + |20-18| + |30-30| = 4
        # |10| + |20| + |30| = 60
        # WMAPE = 4 / 60
        y_true = [10, 20, 30]
        y_pred = [12, 18, 30]
        assert math.isclose(wmape(y_true, y_pred), 4 / 60)

    def test_zero_truth_returns_nan(self) -> None:
        y_true = [0, 0, 0]
        y_pred = [1, 2, 3]
        assert math.isnan(wmape(y_true, y_pred))

    def test_handles_zeros_in_truth(self) -> None:
        # Standard MAPE would fail; WMAPE handles this gracefully
        y_true = [0, 10, 20]
        y_pred = [1, 9, 22]
        # |0-1| + |10-9| + |20-22| = 4
        # |0| + |10| + |20| = 30
        assert math.isclose(wmape(y_true, y_pred), 4 / 30)


class TestBias:
    def test_no_bias(self) -> None:
        y_true = [10, 20, 30]
        y_pred = [10, 20, 30]
        assert bias(y_true, y_pred) == 0.0

    def test_positive_bias(self) -> None:
        # Over-forecasting
        y_true = [10, 10, 10]
        y_pred = [12, 12, 12]
        assert bias(y_true, y_pred) > 0

    def test_negative_bias(self) -> None:
        # Under-forecasting (stockout risk)
        y_true = [10, 10, 10]
        y_pred = [8, 8, 8]
        assert bias(y_true, y_pred) < 0


class TestRMSE:
    def test_perfect_prediction(self) -> None:
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        assert rmse(y_true, y_pred) == 0.0

    def test_known_value(self) -> None:
        y_true = [0, 0, 0]
        y_pred = [1, 1, 1]
        assert math.isclose(rmse(y_true, y_pred), 1.0)


class TestMAE:
    def test_perfect_prediction(self) -> None:
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_known_value(self) -> None:
        # |1-2| + |2-4| + |3-6| = 6, mean = 2
        assert math.isclose(mae([1, 2, 3], [2, 4, 6]), 2.0)


class TestComputeAllMetrics:
    def test_returns_all_keys(self) -> None:
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 30])
        result = compute_all_metrics(y_true, y_pred)
        assert set(result.keys()) == {"wmape", "bias", "rmse", "mae"}
        assert all(isinstance(v, float) for v in result.values())
