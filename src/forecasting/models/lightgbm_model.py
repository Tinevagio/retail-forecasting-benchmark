"""LightGBM forecaster with direct multi-step strategy.

Why direct multi-step (one model per horizon h):
- Simple to debug: each model is independent, predictions don't compound
- Avoids error accumulation typical of recursive forecasting
- Allows different feature sets per horizon if needed (not used here, but
  we keep the option open)

Why LightGBM specifically:
- Handles tabular features natively (no embedding/normalization required)
- Fast: the EDA-motivated feature set generates wide tables and LGBM
  scales well on those
- Robust: handles mixed types, missing values, outliers without preprocessing
- Standard in retail forecasting (M5 winners used LightGBM heavily)

Implementation notes:
- We follow scikit-learn's API conventions internally for clarity
- Predictions are clipped at 0 (negative sales are nonsensical)
- The fit() method handles feature preparation; the user supplies a
  `feature_cols` list to control which columns are used
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Self

import lightgbm as lgb
import numpy as np
import polars as pl

from forecasting.models.base import Forecaster


def _next_periods(last_date: date, horizon: int, frequency: str) -> list[date]:
    """Generate the next `horizon` period start dates after last_date."""
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
    """Direct multi-step LightGBM forecaster.

    Trains one separate LightGBM model per horizon step (h=1, h=2, ..., h=H).
    Each model sees the same features but a different target (the value h
    periods ahead).

    Args:
        feature_cols: List of column names to use as features.
        horizon: Maximum forecast horizon in periods.
        frequency: "W" or "M", drives date arithmetic at predict time.
        target_col: Column containing the target values.
        id_col: Series identifier column.
        date_col: Date column.
        lgb_params: Dict of LightGBM hyperparameters. Sensible defaults
            are applied if None.
        n_estimators: Number of boosting rounds.
    """

    def __init__(
        self,
        feature_cols: list[str],
        horizon: int,
        frequency: str = "W",
        target_col: str = "sales",
        id_col: str = "id",
        date_col: str = "date",
        lgb_params: dict | None = None,
        n_estimators: int = 200,
    ) -> None:
        self.feature_cols = list(feature_cols)
        self.horizon = horizon
        self.frequency = frequency
        self.target_col = target_col
        self.id_col = id_col
        self.date_col = date_col
        self.n_estimators = n_estimators

        # Sensible defaults; phase 3.3 will tune these via Optuna
        self.lgb_params = lgb_params or {
            "objective": "regression_l1",  # MAE-like objective, robust to outliers
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbose": -1,
        }

        # Filled by fit()
        self._models: dict[int, lgb.Booster] = {}
        self._latest_features: pl.DataFrame | None = None
        self._last_date: date | None = None

    @property
    def name(self) -> str:
        return "LightGBMForecaster"

    def _build_targets(self, df: pl.DataFrame) -> pl.DataFrame:
        """Build h-step ahead target columns, one per horizon.

        For horizon h, target_h_col[t] = sales[t + h]. Missing for the last
        h rows of each series, which we drop before training.
        """
        df_sorted = df.sort([self.id_col, self.date_col])
        target_cols = [
            pl.col(self.target_col).shift(-h).over(self.id_col).alias(f"_target_h{h}")
            for h in range(1, self.horizon + 1)
        ]
        return df_sorted.with_columns(target_cols)

    def fit(self, train_df: pl.DataFrame) -> Self:
        if not self.feature_cols:
            raise ValueError("feature_cols must not be empty")

        # Verify all features exist
        missing = set(self.feature_cols) - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing feature columns in train_df: {missing}")

        df_with_targets = self._build_targets(train_df)

        # Train one model per horizon step
        for h in range(1, self.horizon + 1):
            target_col = f"_target_h{h}"
            usable = df_with_targets.filter(pl.col(target_col).is_not_null())
            if usable.height == 0:
                continue

            X = usable.select(self.feature_cols).to_pandas()
            y = usable[target_col].to_numpy()

            train_set = lgb.Dataset(X, label=y)
            booster = lgb.train(
                params=self.lgb_params,
                train_set=train_set,
                num_boost_round=self.n_estimators,
            )
            self._models[h] = booster

        # Cache the most recent observation per series, with its features.
        # At predict time we use these as the basis for the h-step forecasts.
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
            if booster is None:
                # Should only happen if the training data was too short for h
                preds = np.full(latest.height, np.nan)
            else:
                preds = booster.predict(X)
            preds = np.clip(preds, 0.0, None)  # No negative sales
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
        """Return per-feature importance averaged across horizon-specific models.

        Args:
            importance_type: "gain" or "split" (LightGBM convention).

        Returns:
            DataFrame with columns: feature, importance.
        """
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
