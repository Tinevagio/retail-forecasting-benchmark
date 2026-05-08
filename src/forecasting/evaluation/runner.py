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

Predictions persistence (added for project 2)
---------------------------------------------
By default the runner returns metrics only and discards the per-row
predictions. Pass `return_predictions=True` to also get a long-format
predictions dataframe with columns:

    model_name, fold_id, id, date, y_true, y_pred

This is needed by the project 2 stock-optimization workflow, which uses the
point forecasts as a baseline for "point forecast + safety stock" against
the new probabilistic approach. The default (False) preserves the existing
contract — no other caller is affected.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Literal, overload

import polars as pl

from forecasting.data.splits import apply_fold
from forecasting.evaluation.metrics import bias, mae, rmse, wmape

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forecasting.data.splits import Fold
    from forecasting.models.base import Forecaster

# Schema of the predictions dataframe when return_predictions=True.
# Documented as a module constant so callers (scripts/run_*.py, downstream
# project 2 loader) can validate it.
PREDICTIONS_COLUMNS: tuple[str, ...] = (
    "model_name",
    "fold_id",
    "id",
    "date",
    "y_true",
    "y_pred",
)


def _compute_metrics(
    actual: pl.DataFrame,
    predicted: pl.DataFrame,
    join_keys: list[str],
) -> tuple[dict[str, float], pl.DataFrame]:
    """Join actuals and predictions, then compute the metric suite.

    Returns
    -------
    metrics : dict
        wmape / bias / rmse / mae / n_obs.
    merged : pl.DataFrame
        The joined dataframe with columns join_keys + sales + prediction.
        Returned so the caller can also persist per-row predictions if needed,
        without redoing the join.

    Args:
        actual: DataFrame with join_keys + sales.
        predicted: DataFrame with join_keys + prediction.
        join_keys: Columns to join on (typically ["id", "date"]).
    """
    merged = actual.join(predicted, on=join_keys, how="inner")
    if merged.height == 0:
        return (
            {
                "wmape": float("nan"),
                "bias": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "n_obs": 0,
            },
            merged,
        )
    y_true = merged["sales"].to_numpy()
    y_pred = merged["prediction"].to_numpy()
    return (
        {
            "wmape": wmape(y_true, y_pred),
            "bias": bias(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mae": mae(y_true, y_pred),
            "n_obs": int(merged.height),
        },
        merged,
    )


# Overloads keep the typed contract clean: with default False, return is a
# single DataFrame; with True, it is a tuple. Existing callers never see the
# tuple form, so this stays non-breaking.
@overload
def evaluate_model(
    model: Forecaster,
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
    *,
    return_predictions: Literal[False] = False,
) -> pl.DataFrame: ...


@overload
def evaluate_model(
    model: Forecaster,
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
    *,
    return_predictions: Literal[True],
) -> tuple[pl.DataFrame, pl.DataFrame]: ...


def evaluate_model(
    model: Forecaster,
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
    *,
    return_predictions: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """Run walk-forward CV for a single model.

    Args:
        model: A Forecaster instance. A fresh copy is fit per fold.
        df: Long-format DataFrame with id, date, sales.
        folds: Output of `make_walk_forward_folds()`.
        horizon: Number of periods to forecast per fold (typically equal to
            the test window length / period frequency).
        return_predictions: If True, also return per-row predictions across
            all folds (long format). Default False (non-breaking).

    Returns:
        If return_predictions=False (default), a DataFrame with one row per
        fold and metric columns: model_name, fold_id, train_end, test_start,
        test_end, wmape, bias, rmse, mae, n_obs.

        If return_predictions=True, a tuple (metrics_df, predictions_df) where
        predictions_df has columns: model_name, fold_id, id, date, y_true, y_pred.
    """
    rows: list[dict[str, object]] = []
    pred_chunks: list[pl.DataFrame] = []

    for fold in folds:
        train, test = apply_fold(df, fold)
        if train.height == 0 or test.height == 0:
            continue

        # Refit a fresh model per fold to avoid leakage from previous fits
        fold_model = deepcopy(model)
        fold_model.fit(train)

        test_ids = test["id"].unique().to_list()
        predictions = fold_model.predict(horizon=horizon, ids=test_ids)

        metrics, merged = _compute_metrics(
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

        if return_predictions and merged.height > 0:
            pred_chunks.append(
                merged.select(
                    pl.lit(model.name).alias("model_name"),
                    pl.lit(fold.fold_id).alias("fold_id"),
                    pl.col("id"),
                    pl.col("date"),
                    pl.col("sales").alias("y_true"),
                    pl.col("prediction").alias("y_pred"),
                )
            )

    metrics_df = pl.DataFrame(rows)

    if not return_predictions:
        return metrics_df

    predictions_df = (
        pl.concat(pred_chunks)
        if pred_chunks
        else pl.DataFrame(schema=dict.fromkeys(PREDICTIONS_COLUMNS, pl.Utf8))
    )
    return metrics_df, predictions_df


@overload
def evaluate_models(
    models: Sequence[Forecaster],
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
    *,
    return_predictions: Literal[False] = False,
) -> pl.DataFrame: ...


@overload
def evaluate_models(
    models: Sequence[Forecaster],
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
    *,
    return_predictions: Literal[True],
) -> tuple[pl.DataFrame, pl.DataFrame]: ...


def evaluate_models(
    models: Sequence[Forecaster],
    df: pl.DataFrame,
    folds: list[Fold],
    horizon: int,
    *,
    return_predictions: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, pl.DataFrame]:
    """Run walk-forward CV for several models and concatenate results.

    Convenience wrapper around `evaluate_model()` for benchmarks. See that
    function for the meaning of `return_predictions`.
    """
    if not return_predictions:
        all_results = [evaluate_model(m, df, folds, horizon) for m in models]
        return pl.concat(all_results)

    metrics_chunks: list[pl.DataFrame] = []
    pred_chunks: list[pl.DataFrame] = []
    for model in models:
        m_df, p_df = evaluate_model(model, df, folds, horizon, return_predictions=True)
        metrics_chunks.append(m_df)
        pred_chunks.append(p_df)
    return pl.concat(metrics_chunks), pl.concat(pred_chunks)


def aggregate_by_fold(results: pl.DataFrame) -> pl.DataFrame:
    """Average metrics across folds per model.

    Useful for the headline summary table.

    Args:
        results: Output of `evaluate_models()` (metrics dataframe).

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
