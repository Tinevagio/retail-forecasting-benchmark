"""Loading utilities for the M5 Forecasting Accuracy dataset.

The M5 dataset comes in 4 CSVs:
- calendar.csv: date metadata (day of week, events, SNAP days)
- sales_train_evaluation.csv: daily unit sales in wide format (one row per SKU x store)
- sell_prices.csv: weekly prices per SKU x store
- sample_submission.csv: submission format reference

This module provides typed loaders that return either pandas or polars DataFrames,
with sensible dtype defaults to keep memory usage low.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from forecasting.config import M5_RAW_FILES, RAW_DATA_DIR


def _file_path(key: str, data_dir: Path | None = None) -> Path:
    """Resolve the path of an M5 file by key (e.g. 'calendar')."""
    base = data_dir or RAW_DATA_DIR
    if key not in M5_RAW_FILES:
        raise KeyError(f"Unknown M5 file key: {key!r}. Known keys: {list(M5_RAW_FILES)}")
    path = base / M5_RAW_FILES[key]
    if not path.exists():
        raise FileNotFoundError(
            f"M5 file not found: {path}. Did you run `python scripts/download_data.py` ?"
        )
    return path


def load_calendar(data_dir: Path | None = None) -> pl.DataFrame:
    """Load the calendar table.

    Columns:
        date (Date), wm_yr_wk (Int), weekday (Str), wday (Int), month (Int),
        year (Int), d (Str, e.g. 'd_1'), event_name_1, event_type_1,
        event_name_2, event_type_2, snap_CA, snap_TX, snap_WI

    SNAP days are days when food stamp recipients can use their benefits;
    they cause notable demand spikes in food categories.
    """
    path = _file_path("calendar", data_dir)
    return pl.read_csv(path, try_parse_dates=True)


def load_sales(data_dir: Path | None = None) -> pl.DataFrame:
    """Load daily unit sales in wide format.

    Each row is a SKU x store, with columns:
        id, item_id, dept_id, cat_id, store_id, state_id, d_1, d_2, ..., d_1941

    Returns ~30,490 rows x ~1,947 columns.

    For analysis, you typically want this in long format — see `melt_sales()`.
    """
    path = _file_path("sales", data_dir)
    return pl.read_csv(path)


def load_prices(data_dir: Path | None = None) -> pl.DataFrame:
    """Load weekly sell prices per SKU x store.

    Columns: store_id, item_id, wm_yr_wk, sell_price

    Note: prices only appear from the week the product is first sold in a store,
    which is useful for cold-start / new product detection.
    """
    path = _file_path("prices", data_dir)
    return pl.read_csv(path)


def melt_sales(
    sales_wide: pl.DataFrame,
    calendar: pl.DataFrame | None = None,
    drop_zero_tail: bool = False,
) -> pl.DataFrame:
    """Convert wide sales (one column per day) to long format (one row per SKU x day).

    Args:
        sales_wide: Output of `load_sales()`.
        calendar: If provided, joins the date column from calendar onto the result.
        drop_zero_tail: If True, drops rows where sales=0 AND the SKU has had no
            sales yet (typical for products not launched yet in a store). Useful
            for memory but distorts intermittence statistics — keep False for EDA.

    Returns:
        Long DataFrame with columns:
            id, item_id, dept_id, cat_id, store_id, state_id, d, sales,
            and date if calendar is provided.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    long = sales_wide.unpivot(
        index=id_cols,
        on=day_cols,
        variable_name="d",
        value_name="sales",
    )

    if calendar is not None:
        long = long.join(calendar.select(["d", "date"]), on="d", how="left")

    if drop_zero_tail:
        # For each id, find the first day with sales > 0 and drop earlier rows
        first_sale = (
            long.filter(pl.col("sales") > 0)
            .group_by("id")
            .agg(pl.col("d").min().alias("first_sale_d"))
        )
        long = (
            long.join(first_sale, on="id", how="inner")
            .filter(pl.col("d") >= pl.col("first_sale_d"))
            .drop("first_sale_d")
        )

    return long


def load_all(
    data_dir: Path | None = None,
    melt: bool = False,
    backend: Literal["polars", "pandas"] = "polars",
) -> dict:
    """Convenience: load all M5 tables in one call.

    Args:
        data_dir: Override the default data directory.
        melt: If True, returns sales in long format with date column.
        backend: "polars" (default, fast) or "pandas" (familiar).

    Returns:
        Dict with keys "calendar", "sales", "prices".
    """
    calendar = load_calendar(data_dir)
    sales = load_sales(data_dir)
    prices = load_prices(data_dir)

    if melt:
        sales = melt_sales(sales, calendar)

    if backend == "pandas":
        return {
            "calendar": calendar.to_pandas(),
            "sales": sales.to_pandas(),
            "prices": prices.to_pandas(),
        }
    return {"calendar": calendar, "sales": sales, "prices": prices}
