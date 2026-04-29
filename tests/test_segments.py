"""Tests for segment classification utilities."""

from __future__ import annotations

import math

import polars as pl
import pytest

from forecasting.evaluation.segments import (
    add_intermittence_labels,
    classify_intermittence,
    compute_lift,
)


class TestClassifyIntermittence:
    def test_high_volume(self) -> None:
        assert classify_intermittence(0.05) == "high_volume"
        assert classify_intermittence(0.0) == "high_volume"

    def test_medium(self) -> None:
        assert classify_intermittence(0.10) == "medium"
        assert classify_intermittence(0.30) == "medium"
        assert classify_intermittence(0.49) == "medium"

    def test_intermittent(self) -> None:
        assert classify_intermittence(0.50) == "intermittent"
        assert classify_intermittence(0.65) == "intermittent"
        assert classify_intermittence(0.79) == "intermittent"

    def test_extreme(self) -> None:
        assert classify_intermittence(0.80) == "extreme_intermittent"
        assert classify_intermittence(0.95) == "extreme_intermittent"
        assert classify_intermittence(1.0) == "extreme_intermittent"

    def test_invalid_input_raises(self) -> None:
        with pytest.raises(ValueError):
            classify_intermittence(-0.1)
        with pytest.raises(ValueError):
            classify_intermittence(1.5)


class TestAddIntermittenceLabels:
    def test_adds_columns(self) -> None:
        df = pl.DataFrame(
            {
                "id": ["A"] * 10 + ["B"] * 10,
                "sales": [1] * 10 + [0] * 9 + [1],  # A: 0% zeros, B: 90% zeros
            }
        )
        result = add_intermittence_labels(df)
        assert "zero_share" in result.columns
        assert "intermittence_label" in result.columns

    def test_correct_labels(self) -> None:
        df = pl.DataFrame(
            {
                "id": ["A"] * 10 + ["B"] * 10,
                "sales": [1] * 10 + [0] * 9 + [1],
            }
        )
        result = add_intermittence_labels(df)
        # A has 0% zeros -> high_volume
        a = result.filter(pl.col("id") == "A")
        assert a["intermittence_label"][0] == "high_volume"
        # B has 90% zeros -> extreme_intermittent
        b = result.filter(pl.col("id") == "B")
        assert b["intermittence_label"][0] == "extreme_intermittent"


class TestComputeLift:
    def test_positive_lift(self) -> None:
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "A", "A"],
                "on_promo": [False, False, True, True],
                "sales": [10, 10, 15, 15],
            }
        )
        result = compute_lift(df, group_col="cat", condition_col="on_promo")
        # Mean false=10, true=15 -> +50%
        assert math.isclose(result["lift_pct"][0], 50.0, rel_tol=1e-9)

    def test_negative_lift(self) -> None:
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "A", "A"],
                "is_event": [False, False, True, True],
                "sales": [10, 10, 8, 8],
            }
        )
        result = compute_lift(df, group_col="cat", condition_col="is_event")
        # 8/10 - 1 = -20%
        assert math.isclose(result["lift_pct"][0], -20.0, rel_tol=1e-9)

    def test_multi_group(self) -> None:
        df = pl.DataFrame(
            {
                "cat": ["A", "A", "B", "B"],
                "is_event": [False, True, False, True],
                "sales": [10, 12, 10, 5],  # A: +20%, B: -50%
            }
        )
        result = compute_lift(df, group_col="cat", condition_col="is_event")
        result_dict = {row["cat"]: row["lift_pct"] for row in result.iter_rows(named=True)}
        assert math.isclose(result_dict["A"], 20.0, rel_tol=1e-9)
        assert math.isclose(result_dict["B"], -50.0, rel_tol=1e-9)
