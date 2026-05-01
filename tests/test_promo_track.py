"""Tests for the promo features at weekly track level."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from forecasting.features.promo_track import add_promo_features_weekly


def _make_calendar() -> pl.DataFrame:
    """3 weeks (21 days), with wm_yr_wk codes."""
    return pl.DataFrame(
        {
            "d": [f"d_{i}" for i in range(1, 22)],
            "date": [date(2020, 1, 6) + timedelta(days=i) for i in range(21)],
            "wm_yr_wk": [11102] * 7 + [11103] * 7 + [11104] * 7,
        }
    )


def _make_prices() -> pl.DataFrame:
    """One item in one store, weeks 11102-11104. Promo on week 11103 (-10%)."""
    return pl.DataFrame(
        {
            "item_id": ["A"] * 8,
            "store_id": ["S1"] * 8,
            "wm_yr_wk": [11098, 11099, 11100, 11101, 11102, 11103, 11104, 11105],
            "sell_price": [10.0, 10.0, 10.0, 10.0, 10.0, 9.0, 10.0, 10.0],
        }
    )


def _make_track() -> pl.DataFrame:
    week_starts = [date(2020, 1, 6), date(2020, 1, 13), date(2020, 1, 20)]
    return pl.DataFrame(
        {
            "id": ["A__S1"] * 3,
            "date": week_starts,
            "sales": [10.0, 12.0, 11.0],
        }
    )


def _make_hierarchy() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["A__S1"],
            "item_id": ["A"],
            "store_id": ["S1"],
            "state_id": ["CA"],
        }
    )


class TestAddPromoFeaturesWeekly:
    def test_adds_expected_columns(self) -> None:
        track = _make_track()
        prices = _make_prices()
        cal = _make_calendar()
        hier = _make_hierarchy()
        result = add_promo_features_weekly(track, prices, cal, hier, "item_store")
        assert "is_on_promo" in result.columns
        assert "price_relative_to_ref" in result.columns

    def test_default_values_when_no_price_data(self) -> None:
        # Track contains an unknown item with no price history
        track = pl.DataFrame(
            {
                "id": ["UNKNOWN__S1"],
                "date": [date(2020, 1, 13)],
                "sales": [5.0],
            }
        )
        hier = pl.DataFrame(
            {
                "id": ["UNKNOWN__S1"],
                "item_id": ["UNKNOWN"],
                "store_id": ["S1"],
                "state_id": ["CA"],
            }
        )
        result = add_promo_features_weekly(
            track, _make_prices(), _make_calendar(), hier, "item_store"
        )
        # No price -> not on promo, ratio defaults to 1.0
        assert result["is_on_promo"][0] == 0
        assert result["price_relative_to_ref"][0] == 1.0
