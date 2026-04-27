"""Forecasting metrics, with emphasis on business-relevant measures.

Beyond the usual RMSE/MAE, supply chain forecasting cares deeply about:
- WMAPE (Weighted MAPE): handles zeros and weights by sales volume
- Bias: systematic over/under-forecasting drives stockouts or excess inventory
- Service level: probability of satisfying demand at given stock policy
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def wmape(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Weighted Mean Absolute Percentage Error.

    WMAPE = sum(|y_true - y_pred|) / sum(|y_true|)

    Preferred over MAPE because it handles zeros gracefully and naturally
    weights series by their volume — which matches business priorities.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        WMAPE as a float in [0, +inf). Returns NaN if sum of y_true is zero.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_arr).sum()
    if denominator == 0:
        return float("nan")
    return float(np.abs(y_true_arr - y_pred_arr).sum() / denominator)


def bias(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Forecast bias (mean forecast error, normalized).

    bias = mean(y_pred - y_true) / mean(y_true)

    Positive bias means systematic over-forecasting (excess inventory risk),
    negative bias means under-forecasting (stockout risk). This is often
    more actionable than absolute error for replenishment decisions.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denominator = np.abs(y_true_arr).mean()
    if denominator == 0:
        return float("nan")
    return float((y_pred_arr - y_true_arr).mean() / denominator)


def rmse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Root Mean Squared Error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def mae(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean Absolute Error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def compute_all_metrics(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> dict[str, float]:
    """Compute the full standard metric suite at once."""
    return {
        "wmape": wmape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }
