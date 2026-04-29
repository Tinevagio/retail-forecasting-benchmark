"""Tests for the data loading module.

These tests use small fixture DataFrames rather than the full M5 dataset,
so they run fast and don't require the data to be downloaded.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from forecasting.data.load import _file_path, melt_sales


def test_file_path_unknown_key() -> None:
    """Unknown keys should raise KeyError with a helpful message."""
    with pytest.raises(KeyError, match="Unknown M5 file key"):
        _file_path("nonexistent_key")


def test_file_path_missing_file(tmp_path: Path) -> None:
    """Missing files should raise FileNotFoundError pointing to the download script."""
    with pytest.raises(FileNotFoundError, match="download_data"):
        _file_path("calendar", data_dir=tmp_path)


def _make_fake_sales() -> pl.DataFrame:
    """Build a minimal fake sales dataframe in M5 wide format."""
    return pl.DataFrame(
        {
            "id": ["SKU_1_X", "SKU_2_X"],
            "item_id": ["SKU_1", "SKU_2"],
            "dept_id": ["DEPT_A", "DEPT_A"],
            "cat_id": ["CAT_1", "CAT_1"],
            "store_id": ["STORE_X", "STORE_X"],
            "state_id": ["S1", "S1"],
            "d_1": [0, 5],
            "d_2": [3, 0],
            "d_3": [2, 1],
        }
    )


def _make_fake_calendar() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "d": ["d_1", "d_2", "d_3"],
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    ).with_columns(pl.col("date").str.to_date())


class TestMeltSales:
    def test_basic_shape(self) -> None:
        """Long format should have 1 row per (SKU, day)."""
        wide = _make_fake_sales()
        long = melt_sales(wide)
        # 2 SKUs x 3 days = 6 rows
        assert long.height == 6
        # Original ID columns + d + sales = 8 columns
        assert "sales" in long.columns
        assert "d" in long.columns

    def test_sales_values_preserved(self) -> None:
        """The unpivoted sales values must match the original wide values."""
        wide = _make_fake_sales()
        long = melt_sales(wide)
        sku1_d2 = long.filter((pl.col("id") == "SKU_1_X") & (pl.col("d") == "d_2"))["sales"].item()
        assert sku1_d2 == 3

    def test_with_calendar_adds_date(self) -> None:
        wide = _make_fake_sales()
        cal = _make_fake_calendar()
        long = melt_sales(wide, calendar=cal)
        assert "date" in long.columns
        assert long["date"].null_count() == 0

    def test_drop_zero_tail(self) -> None:
        """SKU_1 has 0 sales on d_1 then non-zero — d_1 should be dropped."""
        wide = _make_fake_sales()
        long = melt_sales(wide, drop_zero_tail=True)
        sku1 = long.filter(pl.col("id") == "SKU_1_X")
        # SKU_1: d_1=0 (dropped), d_2=3, d_3=2 → 2 rows
        assert sku1.height == 2
        # SKU_2: d_1=5 (kept), d_2=0 (kept, after first sale), d_3=1 → 3 rows
        sku2 = long.filter(pl.col("id") == "SKU_2_X")
        assert sku2.height == 3
