"""Cross-validation runner for forecasting models.

Orchestrates the full evaluation loop:
1. Walk-forward folds over the dataset
2. For each fold: fit the model on train, predict on test
3. Compute metrics globally and per segment
4. Return a tidy DataFrame for analysis

Why this matters: the same runner is used for naive baselines, Holt-Winters,
LightGBM, and any future model. This guarantees that all comparisons in the
benchmark use identical splits, identical metrics, identical aggregation —
the only thing that varies is the model. Without this, ML "wins" could come
from sloppy evaluation rather than real improvements.
"""

from __future__ import annotations

from copy import deepcopy

import polars as pl

from forecasting.data.splits import Fold, apply_fold
from forecasting.evaluation.metrics import bias, mae, rmse, wmape
from forecasting.models.base import Forecaster


def _compute_metrics(
    actual: pl.DataFrame,
    predicted: pl.DataFrame,
    join_keys: list[str],
) -> dict[str, float]:
    """Join actuals and predictions, then compute the metric suite.

    Args:
        actual: DataFrame with join_keys + sales.
        predicted: DataFrame with join_keys + prediction.
        join_keys: Columns to join on (typically ["id", "date"]).
    """
    merged = actual.join(predicted, on=join_keys, how="inner")
    if merged.height == 0:
        return {
            "wmape": float("nan"),
            "bias": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "n_obs": 0,
        }
    y_true = merged["sales"].to_numpy()
    y_pred = merged["prediction"].to_numpy()
    return {
        "wmape": wmape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n_obs": int(merged.height),
    }


def evaluate_model(
    model: Forecaster,
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
) -> pl.DataFrame:
    """Run walk-forward CV for a single model.

    Args:
        model: A Forecaster instance. A fresh copy is fit per fold.
        df: Long-format DataFrame with id, date, sales.
        folds: Output of `make_walk_forward_folds()`.
        horizon: Number of periods to forecast per fold (typically equal to
            the test window length / period frequency).

    Returns:
        DataFrame with one row per fold and columns:
            model_name, fold_id, train_end, test_start, test_end,
            wmape, bias, rmse, mae, n_obs
    """
    rows = []
    for fold in folds:
        train, test = apply_fold(df, fold)
        if train.height == 0 or test.height == 0:
            continue

        # Refit a fresh model per fold to avoid leakage from previous fits
        fold_model = deepcopy(model)
        fold_model.fit(train)

        test_ids = test["id"].unique().to_list()
        predictions = fold_model.predict(horizon=horizon, ids=test_ids)

        metrics = _compute_metrics(
            actual=test.select(["id", "date", "sales"]),
            predicted=predictions,
            join_keys=["id", "date"],
        )
        rows.append(
            {
                "model_name": model.name,
                "fold_id": fold.fold_id,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                **metrics,
            }
        )

    return pl.DataFrame(rows)


def evaluate_models(
    models: list[Forecaster],
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
) -> pl.DataFrame:
    """Run walk-forward CV for several models and concatenate results.

    Convenience wrapper around `evaluate_model()` for benchmarks.
    """
    all_results = [evaluate_model(m, df, folds, horizon) for m in models]
    return pl.concat(all_results)


def aggregate_by_fold(results: pl.DataFrame) -> pl.DataFrame:
    """Average metrics across folds per model.

    Useful for the headline summary table.

    Args:
        results: Output of `evaluate_models()`.

    Returns:
        DataFrame with one row per model and average metrics.
    """
    return (
        results.group_by("model_name")
        .agg(
            [
                pl.col("wmape").mean().alias("wmape_mean"),
                pl.col("wmape").std().alias("wmape_std"),
                pl.col("bias").mean().alias("bias_mean"),
                pl.col("rmse").mean().alias("rmse_mean"),
                pl.col("mae").mean().alias("mae_mean"),
                pl.col("n_obs").sum().alias("n_obs_total"),
            ]
        )
        .sort("wmape_mean")
    )
