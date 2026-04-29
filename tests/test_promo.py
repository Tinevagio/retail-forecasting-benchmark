"""Tests for promotion detection from prices."""

from __future__ import annotations

import polars as pl

from forecasting.features.promo import detect_promo_from_prices, join_promo_features


def _make_prices() -> pl.DataFrame:
    """One SKU with a clear promo period (week 5 = -10% drop)."""
    return pl.DataFrame(
        {
            "item_id": ["X"] * 10,
            "store_id": ["S1"] * 10,
            "wm_yr_wk": list(range(11101, 11111)),
            "sell_price": [10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )


class TestDetectPromo:
    def test_clear_promo_is_flagged(self) -> None:
        """Week with -10% price (vs 10.0 reference) should be flagged."""
        prices = _make_prices()
        flagged = detect_promo_from_prices(prices, window_size=4, threshold=0.95)
        promo_row = flagged.filter(pl.col("sell_price") == 9.0)
        assert promo_row["on_promo"][0] is True

    def test_no_false_positive_on_stable_price(self) -> None:
        prices = _make_prices()
        flagged = detect_promo_from_prices(prices, window_size=4, threshold=0.95)
        non_promo = flagged.filter(pl.col("sell_price") == 10.0)
        # All stable-price rows should NOT be flagged
        assert not non_promo["on_promo"].any()

    def test_relative_price_is_correct(self) -> None:
        prices = _make_prices()
        flagged = detect_promo_from_prices(prices, window_size=4, threshold=0.95)
        promo_row = flagged.filter(pl.col("sell_price") == 9.0)
        # 9 / 10 = 0.9
        assert abs(promo_row["price_relative_to_ref"][0] - 0.9) < 1e-9

    def test_threshold_strictness(self) -> None:
        """A very strict threshold (0.99) should flag the promo;
        a very lenient one (0.5) should NOT flag a -10% drop."""
        prices = _make_prices()
        strict = detect_promo_from_prices(prices, window_size=4, threshold=0.99)
        lenient = detect_promo_from_prices(prices, window_size=4, threshold=0.50)
        assert strict["on_promo"].sum() >= lenient["on_promo"].sum()


class TestJoinPromoFeatures:
    def test_joins_correctly_with_calendar(self) -> None:
        prices = _make_prices()
        prices_with_promo = detect_promo_from_prices(prices, window_size=4)

        sales = pl.DataFrame(
            {
                "item_id": ["X", "X"],
                "store_id": ["S1", "S1"],
                "d": ["d_1", "d_2"],
                "sales": [5, 6],
            }
        )
        calendar = pl.DataFrame(
            {
                "d": ["d_1", "d_2"],
                "wm_yr_wk": [11105, 11101],  # d_1 = promo week, d_2 = normal week
            }
        )

        result = join_promo_features(sales, prices_with_promo, calendar)
        assert "on_promo" in result.columns
        assert "price_relative_to_ref" in result.columns
        # Order is preserved by item_id/store_id; check the promo flag
        promo_d1 = result.filter(pl.col("d") == "d_1")["on_promo"][0]
        # d_1 is week 11105 which has the price drop
        assert promo_d1 is True

    def test_missing_prices_filled_with_defaults(self) -> None:
        """Days outside the price coverage get on_promo=False, ratio=1.0."""
        prices_with_promo = detect_promo_from_prices(_make_prices(), window_size=4)

        # Sale for an item with no price data
        sales = pl.DataFrame(
            {
                "item_id": ["UNKNOWN"],
                "store_id": ["S1"],
                "d": ["d_1"],
                "sales": [5],
            }
        )
        calendar = pl.DataFrame({"d": ["d_1"], "wm_yr_wk": [11105]})

        result = join_promo_features(sales, prices_with_promo, calendar)
        assert result["on_promo"][0] is False
        assert result["price_relative_to_ref"][0] == 1.0
