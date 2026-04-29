"""Naive baseline forecasters.

These are the floor: any "real" model should beat them by a meaningful
margin. If Holt-Winters can't beat seasonal naive, we have a problem
(usually a tuning bug or a seasonality misconfiguration).

Implemented baselines:
- HistoricalMean: predicts the mean of each series' training history
- SeasonalNaive: predicts the value from `season_length` periods ago
- DriftNaive: linear extrapolation between first and last training points
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Self

import polars as pl

from forecasting.models.base import Forecaster


def _next_periods(last_date: date, horizon: int, frequency: str) -> list[date]:
    """Generate the next `horizon` period start dates after last_date.

    Args:
        last_date: Last date in the training set (already a period start).
        horizon: Number of periods to generate.
        frequency: "W" (7 days) or "M" (~30 days; we use month arithmetic).
    """
    out = []
    if frequency == "W":
        for h in range(1, horizon + 1):
            out.append(last_date + timedelta(days=7 * h))
    elif frequency == "M":
        # Month arithmetic: roll forward keeping day=1 (period start)
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


class HistoricalMean(Forecaster):
    """Predict each series' training-history mean for every future period.

    Embarrassingly simple but useful: any model that can't beat this isn't
    learning anything beyond the level.
    """

    def __init__(self, frequency: str = "W") -> None:
        self.frequency = frequency
        self._means: pl.DataFrame | None = None
        self._last_date: date | None = None

    def fit(self, train_df: pl.DataFrame) -> Self:
        self._means = train_df.group_by("id").agg(pl.col("sales").mean().alias("mean_sales"))
        self._last_date = train_df["date"].max()
        return self

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        if self._means is None or self._last_date is None:
            raise RuntimeError("Call fit() before predict()")

        means = self._means
        if ids is not None:
            means = means.filter(pl.col("id").is_in(ids))

        future_dates = _next_periods(self._last_date, horizon, self.frequency)
        # Cross join: every id x every future date
        return (
            means.join(pl.DataFrame({"date": future_dates}), how="cross")
            .with_columns(pl.col("mean_sales").alias("prediction"))
            .select(["id", "date", "prediction"])
            .sort(["id", "date"])
        )


class SeasonalNaive(Forecaster):
    """Predict the value from `season_length` periods ago.

    For weekly data with season_length=52, predicts the value 52 weeks ago.
    For monthly data with season_length=12, predicts the value from the
    same month last year.

    This is the "you told me it was seasonal, here's the bare-minimum
    seasonal forecast" baseline. It's surprisingly strong on stable
    products with good periodicity.
    """

    def __init__(self, season_length: int, frequency: str = "W") -> None:
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        self.season_length = season_length
        self.frequency = frequency
        self._history: pl.DataFrame | None = None
        self._last_date: date | None = None

    def fit(self, train_df: pl.DataFrame) -> Self:
        # We just store the training series; predictions look back into it
        self._history = train_df.select(["id", "date", "sales"]).clone()
        self._last_date = train_df["date"].max()
        return self

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        if self._history is None or self._last_date is None:
            raise RuntimeError("Call fit() before predict()")

        future_dates = _next_periods(self._last_date, horizon, self.frequency)

        # For each future date, look back season_length periods to find the value
        if self.frequency == "W":
            lookbacks = [d - timedelta(days=7 * self.season_length) for d in future_dates]
        else:  # monthly
            lookbacks = []
            for d in future_dates:
                y = d.year - (self.season_length // 12)
                m = d.month - (self.season_length % 12)
                if m <= 0:
                    m += 12
                    y -= 1
                lookbacks.append(date(y, m, 1))

        date_map = pl.DataFrame({"date": future_dates, "lookback_date": lookbacks})

        history = self._history
        if ids is not None:
            history = history.filter(pl.col("id").is_in(ids))

        # Join history on lookback_date to get the predicted value
        predictions = (
            date_map.join(history, left_on="lookback_date", right_on="date", how="inner")
            .select(["id", "date", pl.col("sales").alias("prediction")])
            .sort(["id", "date"])
        )

        # Series with insufficient history yield no prediction; fill with their
        # mean as a safe fallback (better than NaN propagating downstream)
        all_ids_history = history["id"].unique().to_list()
        target_ids = ids if ids is not None else all_ids_history
        missing_ids = set(target_ids) - set(predictions["id"].unique().to_list())
        if missing_ids:
            fallback_means = (
                history.filter(pl.col("id").is_in(list(missing_ids)))
                .group_by("id")
                .agg(pl.col("sales").mean().alias("mean_sales"))
            )
            fallback = (
                fallback_means.join(pl.DataFrame({"date": future_dates}), how="cross")
                .with_columns(pl.col("mean_sales").alias("prediction"))
                .select(["id", "date", "prediction"])
            )
            predictions = pl.concat([predictions, fallback]).sort(["id", "date"])

        return predictions


class DriftNaive(Forecaster):
    """Linear extrapolation: slope = (last - first) / (n - 1).

    Captures simple trend without seasonality. Useful as a sanity check
    against models that overfit seasonality on actually-trending series.
    """

    def __init__(self, frequency: str = "W") -> None:
        self.frequency = frequency
        self._params: pl.DataFrame | None = None
        self._last_date: date | None = None

    def fit(self, train_df: pl.DataFrame) -> Self:
        # Compute first, last sales per series and the number of observations
        params = (
            train_df.group_by("id")
            .agg(
                [
                    pl.col("sales").first().alias("first_sales"),
                    pl.col("sales").last().alias("last_sales"),
                    pl.col("sales").count().alias("n_obs"),
                ]
            )
            .with_columns(
                pl.when(pl.col("n_obs") > 1)
                .then(
                    (pl.col("last_sales") - pl.col("first_sales"))
                    / (pl.col("n_obs").cast(pl.Float64) - 1)
                )
                .otherwise(0.0)
                .alias("slope")
            )
        )
        self._params = params
        self._last_date = train_df["date"].max()
        return self

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        if self._params is None or self._last_date is None:
            raise RuntimeError("Call fit() before predict()")

        params = self._params
        if ids is not None:
            params = params.filter(pl.col("id").is_in(ids))

        future_dates = _next_periods(self._last_date, horizon, self.frequency)
        # Step h: prediction = last_sales + slope * h
        steps = pl.DataFrame(
            {
                "date": future_dates,
                "step": list(range(1, horizon + 1)),
            }
        )
        return (
            params.join(steps, how="cross")
            .with_columns(
                (pl.col("last_sales") + pl.col("slope") * pl.col("step"))
                .clip(lower_bound=0.0)  # negative sales are nonsensical
                .alias("prediction")
            )
            .select(["id", "date", "prediction"])
            .sort(["id", "date"])
        )
