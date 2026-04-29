"""Time-series cross-validation with walk-forward (expanding-window) splits.

Why walk-forward and not random K-Fold:
A random KFold would put future observations in the training set and past
observations in the test set, which leaks information backwards in time.
This produces optimistic, dishonest metrics. Walk-forward respects the
temporal order: train on past, evaluate on future, repeat.

Why expanding window (vs sliding window):
Both are valid. Expanding gives the model more data each fold, which is
typical of production: you retrain monthly/weekly with all the data you've
ever had. Sliding window simulates a fixed-size training history. We use
expanding here as the production-realistic default.

A `gap` parameter is exposed for cases where features include forward-looking
lookups (e.g. a 28-day lookback feature requires a 28-day gap between train
and test to avoid leakage). Default is 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class Fold:
    """A single train/test fold of a walk-forward CV.

    Attributes:
        fold_id: 0-indexed fold number.
        train_end: Last date INCLUDED in the training set.
        test_start: First date INCLUDED in the test set.
        test_end: Last date INCLUDED in the test set.
    """

    fold_id: int
    train_end: date
    test_start: date
    test_end: date

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_id}: train -> {self.train_end} | "
            f"test {self.test_start} -> {self.test_end}"
        )


def make_walk_forward_folds(
    min_date: date,
    max_date: date,
    n_folds: int,
    test_horizon_days: int,
    gap_days: int = 0,
) -> list[Fold]:
    """Build a list of walk-forward folds covering the dataset's tail.

    The most recent `n_folds * test_horizon_days` days are used as test
    windows. Each fold trains on everything before its test window (minus
    the gap if any).

    Visual layout (n_folds=3, horizon=14, gap=0):
        [-------------- train --------------][test1][test2][test3]
        [---------------- train ------------------][test2][test3]
        [------------------ train ----------------------][test3]

    Args:
        min_date: First date available in the dataset.
        max_date: Last date available in the dataset.
        n_folds: Number of folds.
        test_horizon_days: Length of each test window in days.
        gap_days: Days to skip between train_end and test_start (to avoid
            leakage from forward-looking features). Default 0.

    Returns:
        List of Fold objects, ordered by fold_id (oldest test first).

    Raises:
        ValueError: if the dataset is too short to fit n_folds.
    """
    total_days = (max_date - min_date).days + 1
    needed_days = n_folds * test_horizon_days + gap_days + 1
    if total_days < needed_days:
        raise ValueError(
            f"Dataset has {total_days} days, need at least {needed_days} for "
            f"{n_folds} folds of {test_horizon_days} days with gap={gap_days}"
        )

    from datetime import timedelta

    folds = []
    # The last test window ends on max_date; earlier folds shift back by horizon
    for i in range(n_folds):
        test_end = max_date - timedelta(days=i * test_horizon_days)
        test_start = test_end - timedelta(days=test_horizon_days - 1)
        train_end = test_start - timedelta(days=gap_days + 1)
        folds.append(
            Fold(
                fold_id=n_folds - 1 - i,  # oldest = 0
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return sorted(folds, key=lambda f: f.fold_id)


def apply_fold(
    df: pl.DataFrame,
    fold: Fold,
    date_col: str = "date",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split a long-format DataFrame into (train, test) according to a Fold.

    Args:
        df: Long-format DataFrame with a date column.
        fold: A Fold object defining the time boundaries.
        date_col: Name of the date column.

    Returns:
        Tuple (train_df, test_df). Rows in the gap (if any) are excluded
        from both sets.
    """
    train = df.filter(pl.col(date_col) <= pl.lit(fold.train_end))
    test = df.filter(
        (pl.col(date_col) >= pl.lit(fold.test_start)) & (pl.col(date_col) <= pl.lit(fold.test_end))
    )
    return train, test
