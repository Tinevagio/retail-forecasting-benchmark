"""LightGBM forecaster — phase 3.3 update.

Changes vs phase 3.1 version:
- Tweedie objective option for intermittent demand (default for retail)
- Stockout correction applied to TRAINING data only, evaluated on raw data

The class signature is backward-compatible: existing callers continue to
work with default parameters. New parameters:
- objective: "regression_l1" | "regression_l2" | "tweedie" (default "tweedie")
- target_correction: "none" | "rolling_mean" | "median" (default "none")
- tweedie_variance_power: float in (1, 2), default 1.5 (used only with tweedie)

Why Tweedie:
The Tweedie distribution interpolates between a Poisson (count of events)
and a Gamma (positive continuous values), making it suited to data that
mixes many zeros with positive continuous values — exactly retail demand.
The variance_power parameter controls the mixture; 1.5 is a common starting
point for retail.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Self

import lightgbm as lgb
import numpy as np
import polars as pl

from forecasting.features.stockout_correction import (
    CorrectionStrategy,
    correct_stockouts,
)
from forecasting.models.base import Forecaster


def _next_periods(last_date: date, horizon: int, frequency: str) -> list[date]:
    out = []
    if frequency == "W":
        for h in range(1, horizon + 1):
            out.append(last_date + timedelta(days=7 * h))
    elif frequency == "M":
        y, m = last_date.year, last_date.month
        for _ in range(horizon):
            m += 1
            if m > 12:
                m = 1
                y += 1
            out.append(date(y, m, 1))
    else:
        raise ValueError(f"Unsupported frequency {frequency!r}")
    return out


class LightGBMForecaster(Forecaster):
    """Direct multi-step LightGBM forecaster with Tweedie + stockout support.

    Args:
        feature_cols: Feature column names to use.
        horizon: Maximum forecast horizon in periods.
        frequency: "W" or "M".
        target_col, id_col, date_col: Column names.
        objective: LightGBM objective. "tweedie" recommended for retail data.
        target_correction: Strategy to clean training-time target. The test
            target is never modified (we evaluate against reality).
        tweedie_variance_power: Only used when objective="tweedie".
        lgb_params: Override hyperparameters; merged on top of defaults.
        n_estimators: Number of boosting rounds.

    Attributes after fit:
        _models: dict[h] -> lightgbm.Booster
        correction_stats: dict with stockout-correction statistics
    """

    def __init__(
        self,
        feature_cols: list[str],
        horizon: int,
        frequency: str = "W",
        target_col: str = "sales",
        id_col: str = "id",
        date_col: str = "date",
        objective: str = "tweedie",
        target_correction: CorrectionStrategy = "none",
        tweedie_variance_power: float = 1.5,
        lgb_params: dict | None = None,
        n_estimators: int = 200,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.horizon = horizon
        self.frequency = frequency
        self.target_col = target_col
        self.id_col = id_col
        self.date_col = date_col
        self.objective = objective
        self.target_correction = target_correction
        self.tweedie_variance_power = tweedie_variance_power
        self.n_estimators = n_estimators

        # Build params dict respecting the caller's overrides
        defaults = {
            "objective": objective,
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbose": -1,
        }
        if objective == "tweedie":
            defaults["tweedie_variance_power"] = tweedie_variance_power
        self.lgb_params = {**defaults, **(lgb_params or {})}

        # Filled by fit()
        self._models: dict[int, lgb.Booster] = {}
        self._latest_features: pl.DataFrame | None = None
        self._last_date: date | None = None
        self.correction_stats: dict[str, float] = {}

    @property
    def name(self) -> str:
        # Provide an informative name for results tables
        suffix_parts = []
        if self.objective != "regression_l1":
            suffix_parts.append(self.objective)
        if self.target_correction != "none":
            suffix_parts.append(f"corr-{self.target_correction}")
        if suffix_parts:
            return f"LightGBM-{'-'.join(suffix_parts)}"
        return "LightGBMForecaster"

    def _build_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        df_sorted = df.sort([self.id_col, self.date_col])
        target_cols = [
            pl.col(self.target_col).shift(-h).over(self.id_col).alias(f"_target_h{h}")
            for h in range(1, self.horizon + 1)
        ]
        return df_sorted.with_columns(target_cols)

    def fit(self, train_df: pl.DataFrame) -> Self:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty")
        missing = set(self.feature_cols) - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing feature columns in train_df: {missing}")

        # Apply stockout correction (training-only)
        if self.target_correction != "none":
            corrected_df, stats = correct_stockouts(
                train_df,
                strategy=self.target_correction,
                target_col=self.target_col,
                id_col=self.id_col,
                date_col=self.date_col,
            )
            self.correction_stats = stats
            train_df = corrected_df
        else:
            self.correction_stats = {
                "n_total_rows": train_df.height,
                "n_suspicious": 0,
                "share_suspicious": 0.0,
                "mean_imputed_value": float("nan"),
            }

        df_with_targets = self._build_targets(train_df)

        for h in range(1, self.horizon + 1):
            target_col_h = f"_target_h{h}"
            usable = df_with_targets.filter(pl.col(target_col_h).is_not_null())
            if usable.height == 0:
                continue

            X = usable.select(self.feature_cols).to_pandas()
            y = usable[target_col_h].to_numpy()

            train_set = lgb.Dataset(X, label=y)
            booster = lgb.train(
                params=self.lgb_params,
                train_set=train_set,
                num_boost_round=self.n_estimators,
            )
            self._models[h] = booster

        self._latest_features = (
            train_df.sort([self.id_col, self.date_col])
            .group_by(self.id_col, maintain_order=True)
            .last()
        )
        self._last_date = train_df[self.date_col].max()
        return self

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        if not self._models or self._latest_features is None or self._last_date is None:
            raise RuntimeError("Call fit() before predict()")
        if horizon > self.horizon:
            raise ValueError(
                f"Requested horizon {horizon} exceeds the trained horizon "
                f"{self.horizon}. Refit with a larger horizon."
            )

        latest = self._latest_features
        if ids is not None:
            latest = latest.filter(pl.col(self.id_col).is_in(ids))

        if latest.height == 0:
            return pl.DataFrame(
                schema={
                    self.id_col: pl.String,
                    self.date_col: pl.Date,
                    "prediction": pl.Float64,
                }
            )

        future_dates = _next_periods(self._last_date, horizon, self.frequency)

        X = latest.select(self.feature_cols).to_pandas()
        rows = []
        for h in range(1, horizon + 1):
            booster = self._models.get(h)
            preds = np.full(latest.height, np.nan) if booster is None else booster.predict(X)
            preds = np.clip(preds, 0.0, None)
            for sku_id, pred_val in zip(latest[self.id_col].to_list(), preds, strict=True):
                rows.append(
                    {
                        self.id_col: sku_id,
                        self.date_col: future_dates[h - 1],
                        "prediction": float(pred_val),
                    }
                )

        return pl.DataFrame(rows).sort([self.id_col, self.date_col])

    def feature_importance(self, importance_type: str = "gain") -> pl.DataFrame:
        if not self._models:
            raise RuntimeError("Call fit() before feature_importance()")

        importances = np.zeros(len(self.feature_cols))
        for booster in self._models.values():
            importances += booster.feature_importance(importance_type=importance_type)
        importances /= len(self._models)

        return pl.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": importances,
            }
        ).sort("importance", descending=True)
