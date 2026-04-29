"""Base interface for all forecasting models in the project.

Every concrete model (naive, Holt-Winters, LightGBM, deep) implements this
interface, so the evaluation runner can treat them uniformly.

Design notes:
- We intentionally keep the interface minimal. Anything model-specific
  (hyperparameters, loss functions, etc.) goes in the constructor.
- `fit` returns self to allow chaining.
- `predict` takes a horizon (number of periods ahead) and a list of series
  ids to forecast. This matches how `statsforecast` and most scalable
  forecasting libraries work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import polars as pl


class Forecaster(ABC):
    """Abstract base class for all forecasting models."""

    @abstractmethod
    def fit(self, train_df: pl.DataFrame) -> Self:
        """Fit the model on a long-format training DataFrame.

        Expected columns:
            - id (str): series identifier
            - date (Date): period start date
            - sales (numeric): target

        Returns:
            self, for chaining.
        """
        ...

    @abstractmethod
    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        """Produce forecasts for the next `horizon` periods.

        Args:
            horizon: Number of periods (days/weeks/months) to forecast ahead.
            ids: Subset of series ids to forecast. If None, forecast all
                series seen during fit.

        Returns:
            Long-format DataFrame with columns:
                - id (str)
                - date (Date): the forecasted period start date
                - prediction (numeric)
        """
        ...

    @property
    def name(self) -> str:
        """Short name for logging and result tables."""
        return self.__class__.__name__
