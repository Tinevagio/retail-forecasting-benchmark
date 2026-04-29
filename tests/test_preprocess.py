"""Tests for the preprocessing utilities."""

from __future__ import annotations

import polars as pl

from forecasting.data.preprocess import (
    detect_suspicious_zeros,
    stockout_summary_by_sku,
)


def _make_fixture() -> pl.DataFrame:
    """Two SKUs over 30 days with controlled zero patterns."""
    dates = [f"2020-01-{d:02d}" for d in range(1, 31)]
    # SKU_A: regular sales of 5, with one zero at day 15 (suspicious)
    sku_a_sales = [5] * 14 + [0] + [5] * 15
    # SKU_B: long stretch of zeros at the start (cold start);
    # zeros far from any sales should NOT be flagged.
    sku_b_sales = [0] * 20 + [3] * 10
    return pl.DataFrame(
        {
            "id": ["SKU_A"] * 30 + ["SKU_B"] * 30,
            "date": (dates + dates),
            "sales": sku_a_sales + sku_b_sales,
        }
    ).with_columns(pl.col("date").str.to_date())


class TestDetectSuspiciousZeros:
    def test_isolated_zero_is_flagged(self) -> None:
        """A zero surrounded by normal sales should be flagged."""
        df = _make_fixture()
        flagged = detect_suspicious_zeros(df)
        # On SKU_A day 15, sales=0 and rolling_mean is high -> suspicious
        sku_a_day15 = flagged.filter(
            (pl.col("id") == "SKU_A") & (pl.col("date") == pl.lit("2020-01-15").str.to_date())
        )
        assert sku_a_day15["suspicious_zero"][0] is True

    def test_cold_start_zeros_not_flagged(self) -> None:
        """Zeros far from any sales (early cold-start) should NOT be flagged.

        Note: zeros NEAR the transition to sales may be flagged because the
        centered rolling mean catches the upcoming activity. This is an
        accepted limitation of the heuristic. We only test the early zeros.
        """
        df = _make_fixture()
        flagged = detect_suspicious_zeros(df, window_size=15)
        # SKU_B early zeros (days 1-12): far enough from sales, should not be flagged
        early_zeros = flagged.filter(
            (pl.col("id") == "SKU_B") & (pl.col("date") <= pl.lit("2020-01-12").str.to_date())
        )
        assert not early_zeros["suspicious_zero"].fill_null(False).any()

    def test_normal_sales_never_flagged(self) -> None:
        """Non-zero days are never flagged regardless of context."""
        df = _make_fixture()
        flagged = detect_suspicious_zeros(df)
        non_zeros = flagged.filter(pl.col("sales") > 0)
        assert not non_zeros["suspicious_zero"].any()

    def test_threshold_parameter(self) -> None:
        """Higher threshold should reduce flagged count."""
        df = _make_fixture()
        flagged_low = detect_suspicious_zeros(df, threshold=0.5)
        flagged_high = detect_suspicious_zeros(df, threshold=10.0)
        assert flagged_low["suspicious_zero"].sum() >= flagged_high["suspicious_zero"].sum()


class TestStockoutSummary:
    def test_aggregates_per_series(self) -> None:
        df = _make_fixture()
        flagged = detect_suspicious_zeros(df)
        summary = stockout_summary_by_sku(flagged)
        assert summary.height == 2  # one row per SKU
        assert {"n_suspicious", "n_zeros", "avg_sales", "suspicious_share"} <= set(summary.columns)

    def test_zero_division_handled(self) -> None:
        """SKUs with no zeros must not produce NaN/Inf in suspicious_share."""
        df = pl.DataFrame(
            {
                "id": ["X"] * 5,
                "date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
                "sales": [1, 2, 3, 4, 5],
            }
        ).with_columns(pl.col("date").str.to_date())
        flagged = detect_suspicious_zeros(df)
        summary = stockout_summary_by_sku(flagged)
        # No zeros -> share should be 0, not NaN
        assert summary["suspicious_share"][0] == 0.0
