"""Lightweight feature store for serving predictions.

In production, prediction-time feature computation is the #1 source of
training/serving skew. The standard solution is a feature store: at
training time, save the last computed features per series. At serving
time, load them directly instead of recomputing.

For this project we use a simple Parquet file on disk. Key per row =
series id; columns = the feature columns the model expects + the date
of those features.

Why Parquet:
- Compact (columnar, compressed)
- Fast to load
- Polars reads it natively
- A real feature store (Feast, Tecton, etc.) is overkill for this scale
  but the abstractions are similar
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def save_feature_store(
    df: pl.DataFrame,
    path: str | Path,
    id_col: str = "id",
    date_col: str = "date",
    feature_cols: list[str] | None = None,
) -> Path:
    """Save the latest feature row per series to a Parquet file.

    Args:
        df: Long-format DataFrame with id, date, feature columns.
        path: Output path (.parquet by convention).
        id_col, date_col: Column names.
        feature_cols: Subset of columns to keep. If None, keep all
            columns except id/date that aren't the target.

    Returns:
        Path the file was written to.
    """
    # Pick the latest row per series (by date)
    latest = df.sort([id_col, date_col]).group_by(id_col, maintain_order=True).last()

    if feature_cols is not None:
        cols_to_keep = [id_col, date_col, *feature_cols]
        missing = set(cols_to_keep) - set(latest.columns)
        if missing:
            raise ValueError(f"Missing columns in df: {missing}")
        latest = latest.select(cols_to_keep)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    latest.write_parquet(p)
    return p


def load_feature_store(path: str | Path) -> pl.DataFrame:
    """Load a feature store back into a Polars DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        DataFrame with one row per series (latest features).

    Raises:
        FileNotFoundError: if the file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No feature store at {p}")
    return pl.read_parquet(p)


def get_features_for_ids(
    feature_store: pl.DataFrame,
    ids: list[str],
    id_col: str = "id",
) -> pl.DataFrame:
    """Filter the feature store to a list of ids.

    Args:
        feature_store: DataFrame loaded by `load_feature_store`.
        ids: List of series ids to retrieve.
        id_col: Identifier column name.

    Returns:
        Subset of the feature store for the requested ids. Order matches
        the input list (ids missing from the store are silently dropped).
    """
    return feature_store.filter(pl.col(id_col).is_in(ids))
