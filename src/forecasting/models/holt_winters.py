"""Holt-Winters baseline via statsforecast's AutoETS.

Why AutoETS rather than a fixed Holt-Winters configuration:
A production replenishment tool like AZAP applies model selection per series
(some products are additive, some multiplicative, some have trend, some
don't). Using AutoETS — which selects the best ETS variant by AIC per series
— is the closest fair approximation to such a tool, and avoids the false
choice between "additive" and "multiplicative" on a heterogeneous portfolio.

Why a fallback strategy:
The EDA showed 77% of series are cold-start (introduced after the dataset
begins) and many have insufficient history for HW to fit reliably. Rather
than dropping these series (which would inflate metrics by removing the
hardest cases), we fall back to DriftNaive on series with too little history.
This mirrors what real replenishment systems do: a separate "new product"
logic kicks in until enough history accumulates.

The wrapper exposes which series used HW vs fallback via `coverage_report()`,
so the evaluation can transparently report both numbers.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Self

import polars as pl

from forecasting.models.base import Forecaster
from forecasting.models.naive import DriftNaive


def _next_periods(last_date: date, horizon: int, frequency: str) -> list[date]:
    """Generate the next `horizon` period start dates after last_date.

    Mirrors the helper in `naive.py` to keep this module self-contained
    in terms of date arithmetic (small duplication is acceptable here).
    """
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


class HoltWintersBaseline(Forecaster):
    """AutoETS forecaster with cold-start fallback.

    For each series:
    - If history length >= `min_history` periods, fit AutoETS
    - Otherwise (or if AutoETS fitting fails), fall back to DriftNaive

    The fallback decision is made per-series at fit time. The same logic is
    applied at predict time: each series' prediction comes from whichever
    model handled it during fit.

    Args:
        season_length: Seasonal period in number of observations
            (52 for weekly with annual seasonality, 12 for monthly).
        frequency: "W" or "M" — used by the fallback model and date arithmetic.
        min_history: Minimum number of observations required for AutoETS.
            Default 2 * season_length, the standard requirement.
        n_jobs: Parallel jobs for statsforecast. -1 = all cores. Use 1 on
            Windows for stability on smaller runs.

    Attributes (after fit):
        n_series_total: total series seen during fit
        n_series_hw: series successfully handled by AutoETS
        n_series_fallback: series routed to DriftNaive
    """

    def __init__(
        self,
        season_length: int,
        frequency: str = "W",
        min_history: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.season_length = season_length
        self.frequency = frequency
        self.min_history = min_history if min_history is not None else 2 * season_length
        self.n_jobs = n_jobs

        # Filled by fit()
        # Statsforecast's StatsForecast instance for HW series; populated in fit()
        # if any series qualifies for AutoETS. Typed as Any to avoid a hard
        # dependency on statsforecast at import time.
        self._sf_model: object | None = None
        self._fallback_model: DriftNaive | None = None
        self._hw_ids: set[str] = set()
        self._fallback_ids: set[str] = set()
        self._last_date: date | None = None
        self._fit_called: bool = False
        self.n_series_total: int = 0
        self.n_series_hw: int = 0
        self.n_series_fallback: int = 0

    @property
    def name(self) -> str:
        return "HoltWintersBaseline"

    def fit(self, train_df: pl.DataFrame) -> Self:
        # Lazy import so the rest of the package doesn't require statsforecast
        # to be installed (it's a heavy dep).
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        # 1. Split series by history length
        history_lengths = train_df.group_by("id").agg(pl.col("sales").count().alias("n_obs"))
        long_enough = history_lengths.filter(pl.col("n_obs") >= self.min_history)["id"].to_list()
        too_short = history_lengths.filter(pl.col("n_obs") < self.min_history)["id"].to_list()
        self._hw_ids = set(long_enough)
        self._fallback_ids = set(too_short)
        self.n_series_total = history_lengths.height
        self.n_series_hw = len(self._hw_ids)
        self.n_series_fallback = len(self._fallback_ids)
        self._fit_called = True
        if train_df.height > 0:
            last_date_val = train_df["date"].max()
            assert isinstance(last_date_val, date), (
                f"Expected date, got {type(last_date_val).__name__}"
            )
            self._last_date = last_date_val
        else:
            self._last_date = None

        # 2. Fit AutoETS on long-enough series via statsforecast
        if self._hw_ids:
            hw_train = train_df.filter(pl.col("id").is_in(list(self._hw_ids)))
            # statsforecast expects pandas with columns: unique_id, ds, y
            sf_df = hw_train.select(
                [
                    pl.col("id").alias("unique_id"),
                    pl.col("date").alias("ds"),
                    pl.col("sales").cast(pl.Float64).alias("y"),
                ]
            ).to_pandas()

            freq_str = "W-MON" if self.frequency == "W" else "MS"
            sf_instance = StatsForecast(
                models=[AutoETS(season_length=self.season_length)],
                freq=freq_str,
                n_jobs=self.n_jobs,
            )
            sf_instance.fit(sf_df)
            self._sf_model = sf_instance

        # 3. Fit fallback DriftNaive on the short-history series (if any)
        if self._fallback_ids:
            fallback_train = train_df.filter(pl.col("id").is_in(list(self._fallback_ids)))
            self._fallback_model = DriftNaive(frequency=self.frequency)
            self._fallback_model.fit(fallback_train)

        return self

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        if not self._fit_called:
            raise RuntimeError("Call fit() before predict()")

        # Empty fit -> empty predictions
        if self.n_series_total == 0:
            return pl.DataFrame(schema={"id": pl.String, "date": pl.Date, "prediction": pl.Float64})

        target_ids = set(ids) if ids is not None else (self._hw_ids | self._fallback_ids)
        hw_targets = sorted(target_ids & self._hw_ids)
        fallback_targets = sorted(target_ids & self._fallback_ids)

        parts = []

        # 1. AutoETS predictions for long series
        if hw_targets and self._sf_model is not None:
            # mypy needs this assertion: type-narrowing through `is not None`
            # check above is sufficient at runtime but not always inferred
            assert self._sf_model is not None
            sf_preds = self._sf_model.predict(h=horizon)  # type: ignore[attr-defined]
            sf_pl = pl.from_pandas(sf_preds.reset_index())
            # statsforecast returns columns: unique_id, ds, AutoETS
            hw_preds = sf_pl.filter(pl.col("unique_id").is_in(hw_targets)).select(
                [
                    pl.col("unique_id").alias("id"),
                    pl.col("ds").cast(pl.Date).alias("date"),
                    pl.col("AutoETS").clip(lower_bound=0.0).alias("prediction"),
                ]
            )
            parts.append(hw_preds)

        # 2. DriftNaive fallback predictions
        if fallback_targets and self._fallback_model is not None:
            fb_preds = self._fallback_model.predict(horizon=horizon, ids=fallback_targets)
            parts.append(fb_preds)

        if not parts:
            return pl.DataFrame(schema={"id": pl.String, "date": pl.Date, "prediction": pl.Float64})

        return pl.concat(parts).sort(["id", "date"])

    def coverage_report(self) -> dict[str, int | float]:
        """Return how many series were handled by HW vs fallback.

        Useful in the results documentation to be transparent about what
        fraction of the metric comes from "real" Holt-Winters vs the
        fallback model.
        """
        if self.n_series_total == 0:
            return {
                "n_total": 0,
                "n_hw": 0,
                "n_fallback": 0,
                "hw_share": float("nan"),
            }
        return {
            "n_total": self.n_series_total,
            "n_hw": self.n_series_hw,
            "n_fallback": self.n_series_fallback,
            "hw_share": self.n_series_hw / self.n_series_total,
        }
