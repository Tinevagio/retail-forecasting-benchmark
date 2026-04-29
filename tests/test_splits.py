"""Tests for time-series cross-validation splits."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import polars as pl
import pytest

from forecasting.data.splits import Fold, apply_fold, make_walk_forward_folds


class TestMakeFolds:
    def test_basic_count(self) -> None:
        folds = make_walk_forward_folds(
            min_date=date(2020, 1, 1),
            max_date=date(2020, 12, 31),
            n_folds=4,
            test_horizon_days=14,
        )
        assert len(folds) == 4
        # Folds are 0-indexed in chronological order (oldest first)
        assert [f.fold_id for f in folds] == [0, 1, 2, 3]

    def test_no_overlap_no_leakage(self) -> None:
        """Train must end before test starts; no overlap."""
        folds = make_walk_forward_folds(
            min_date=date(2020, 1, 1),
            max_date=date(2020, 12, 31),
            n_folds=4,
            test_horizon_days=14,
        )
        for f in folds:
            assert f.train_end < f.test_start
            assert f.test_start <= f.test_end

    def test_test_windows_are_contiguous(self) -> None:
        """Successive test windows should not overlap and cover horizon-days each."""
        folds = make_walk_forward_folds(
            min_date=date(2020, 1, 1),
            max_date=date(2020, 12, 31),
            n_folds=3,
            test_horizon_days=14,
        )
        # Each test window is 14 days
        for f in folds:
            length = (f.test_end - f.test_start).days + 1
            assert length == 14
        # Each fold's test_start is exactly 14 days after the previous fold's
        for prev, nxt in pairwise(folds):
            assert (nxt.test_start - prev.test_start).days == 14

    def test_gap_is_respected(self) -> None:
        folds = make_walk_forward_folds(
            min_date=date(2020, 1, 1),
            max_date=date(2020, 12, 31),
            n_folds=4,
            test_horizon_days=14,
            gap_days=7,
        )
        for f in folds:
            assert (f.test_start - f.train_end).days == 8  # gap=7 + 1 day inclusive

    def test_raises_on_too_short(self) -> None:
        with pytest.raises(ValueError, match="Dataset has"):
            make_walk_forward_folds(
                min_date=date(2020, 1, 1),
                max_date=date(2020, 1, 5),  # 5 days only
                n_folds=4,
                test_horizon_days=14,
            )


class TestApplyFold:
    def test_splits_correctly(self) -> None:
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(30)],
                "id": ["X"] * 30,
                "sales": list(range(30)),
            }
        )
        fold = Fold(
            fold_id=0,
            train_end=date(2020, 1, 20),
            test_start=date(2020, 1, 21),
            test_end=date(2020, 1, 25),
        )
        train, test = apply_fold(df, fold)
        assert train.height == 20  # Jan 1 to Jan 20 inclusive
        assert test.height == 5  # Jan 21 to Jan 25 inclusive
        assert train["date"].max() == fold.train_end
        assert test["date"].min() == fold.test_start
        assert test["date"].max() == fold.test_end

    def test_gap_excludes_rows_between(self) -> None:
        df = pl.DataFrame(
            {
                "date": [date(2020, 1, 1) + timedelta(days=i) for i in range(30)],
                "id": ["X"] * 30,
                "sales": list(range(30)),
            }
        )
        fold = Fold(
            fold_id=0,
            train_end=date(2020, 1, 15),
            test_start=date(2020, 1, 21),  # 5-day gap
            test_end=date(2020, 1, 25),
        )
        train, test = apply_fold(df, fold)
        # Rows on Jan 16-20 are in neither set
        assert train["date"].max() < fold.test_start
        assert test["date"].min() > fold.train_end
