"""Tests for stockout detection and correction."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.features.stockout_correction import (
    correct_stockouts,
    detect_suspicious_zeros,
)


def _make_clean_series(weeks: int = 20, value: float = 10.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["X"] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": [value] * weeks,
        }
    )


def _make_series_with_suspect_zero(
    weeks: int = 20, zero_at_idx: int = 15, normal_value: float = 10.0
) -> pl.DataFrame:
    sales = [normal_value] * weeks
    sales[zero_at_idx] = 0.0
    return pl.DataFrame(
        {
            "id": ["X"] * weeks,
            "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(weeks)],
            "sales": sales,
        }
    )


class TestDetectSuspiciousZeros:
    def test_clean_series_no_flags(self) -> None:
        """A series with no zeros should have nothing flagged."""
        df = _make_clean_series()
        result = detect_suspicious_zeros(df)
        assert result["is_suspicious_zero"].sum() == 0

    def test_zero_after_steady_history_is_flagged(self) -> None:
        """A zero appearing after solid history of activity is flagged."""
        df = _make_series_with_suspect_zero(weeks=20, zero_at_idx=15)
        result = detect_suspicious_zeros(df)
        # The zero at index 15 should be flagged
        flagged = result.filter(pl.col("is_suspicious_zero"))
        assert flagged.height == 1
        assert flagged["sales"][0] == 0.0

    def test_early_zero_not_flagged(self) -> None:
        """Zeros in the first few periods (cold-start) are not flagged."""
        df = _make_series_with_suspect_zero(weeks=20, zero_at_idx=2)
        result = detect_suspicious_zeros(df, min_history=8)
        # Zero too early -> not flagged
        assert result["is_suspicious_zero"].sum() == 0

    def test_zero_after_flat_zeros_not_flagged(self) -> None:
        """If recent history is also zeros, the new zero isn't anomalous."""
        sales = [10.0] * 5 + [0.0] * 15  # series died
        df = pl.DataFrame(
            {
                "id": ["X"] * 20,
                "date": [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(20)],
                "sales": sales,
            }
        )
        result = detect_suspicious_zeros(df)
        # Once the series is mostly zeros, additional zeros aren't flagged
        # because the rolling mean is 0
        n_flagged = int(result["is_suspicious_zero"].sum())
        # Only the very first zeros after the active period might be flagged
        assert n_flagged <= 4  # tolerance for the boundary


class TestCorrectStockouts:
    def test_none_strategy_unchanged(self) -> None:
        df = _make_series_with_suspect_zero()
        corrected, stats = correct_stockouts(df, strategy="none")
        assert corrected.equals(df)
        assert stats["n_suspicious"] == 0

    def test_rolling_mean_replaces_suspicious_zero(self) -> None:
        df = _make_series_with_suspect_zero(weeks=20, zero_at_idx=15)
        corrected, stats = correct_stockouts(df, strategy="rolling_mean")
        # The suspicious zero should now have a positive value (the rolling mean)
        replaced_value = corrected["sales"][15]
        assert replaced_value > 0
        # And it should be close to the normal value (10.0) since recent history is all 10s
        assert abs(replaced_value - 10.0) < 0.5
        assert stats["n_suspicious"] == 1

    def test_other_zeros_in_clean_part_not_modified(self) -> None:
        """Series without flagged zeros should be untouched."""
        df = _make_clean_series()
        corrected, stats = correct_stockouts(df, strategy="rolling_mean")
        assert corrected["sales"].sum() == df["sales"].sum()
        assert stats["n_suspicious"] == 0

    def test_median_strategy(self) -> None:
        df = _make_series_with_suspect_zero(weeks=20, zero_at_idx=15)
        corrected, _ = correct_stockouts(df, strategy="median")
        # Replacement should be the non-zero median = 10.0
        assert corrected["sales"][15] == 10.0
