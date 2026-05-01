"""Stockout correction: imputing suspicious zeros in the training target.

The EDA identified that ~8.2% of zero-sales weeks are likely stockouts
(temporary supply ruptures) rather than genuine zero demand. Training a
model on these polluted zeros teaches it to systematically under-forecast.

Strategy: identify suspicious zeros and replace them with a plausible
estimate based on each series' recent activity. Two approaches available:

1. **rolling_mean** (default): replace suspicious zeros with the rolling
   mean of the surrounding 4 weeks (excluding the suspicious zero itself).
   This is simple, defensible, and reflects "what sales would have looked
   like without the rupture". Used by Walmart's own M5 winner.

2. **median**: replace with the historical non-zero median of the series.
   More conservative, doesn't react to local trends. Better for very
   intermittent series where rolling means are noisy.

CRITICAL DESIGN PRINCIPLE — only correct the TRAINING data, never the test:
- Training: corrected target -> model learns "what sales should be"
- Test: brute target -> we evaluate against reality (rupture or not)
This avoids inflating evaluation metrics: a model that "fixes" stockouts
in evaluation isn't actually better, it's just measuring against a
fictional target.
"""

from __future__ import annotations

from typing import Literal

import polars as pl

CorrectionStrategy = Literal["none", "rolling_mean", "median"]


def detect_suspicious_zeros(
    df: pl.DataFrame,
    target_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
    rolling_window: int = 4,
    threshold_ratio: float = 0.5,
    min_history: int = 8,
) -> pl.DataFrame:
    """Flag zero-sales rows that are likely stockouts.

    A zero is flagged "suspicious" when:
    - The value at this period is exactly 0
    - The series has at least `min_history` periods of history
    - The rolling mean of the previous `rolling_window` periods is
      meaningfully positive (above 0)
    - The historical mean of the series is at least `threshold_ratio` * the
      rolling mean — i.e. this series typically sells, so the zero is
      anomalous compared to recent activity.

    Args:
        df: Long-format DataFrame, sorted by (id, date) or sortable.
        target_col: Column to inspect.
        id_col: Series identifier.
        date_col: Date column for sorting.
        rolling_window: Number of past periods to compute the rolling mean.
        threshold_ratio: Series-mean / rolling-mean ratio above which a zero
            is flagged. 0.5 means "the rolling mean must be at least half
            of the historical mean for a zero to be flagged" — protects
            against series that legitimately drop to 0 after end of life.
        min_history: Minimum number of periods before any flagging happens
            (avoids over-flagging cold-start series).

    Returns:
        DataFrame with an added `is_suspicious_zero` (Bool) column.
    """
    df_sorted = df.sort([id_col, date_col])
    return (
        df_sorted.with_columns(
            [
                # Rolling mean of the previous `rolling_window` periods (excluding self)
                pl.col(target_col)
                .rolling_mean(window_size=rolling_window, min_samples=2)
                .shift(1)
                .over(id_col)
                .alias("_rolling_mean_pre"),
                # Series-level historical mean (computed naively here; in a leakage-
                # sensitive setting, this would also be a rolling/expanding stat)
                pl.col(target_col).mean().over(id_col).alias("_series_mean"),
                # Position in the series (for min_history check)
                pl.col(target_col).cum_count().over(id_col).alias("_position"),
            ]
        )
        .with_columns(
            (
                (pl.col(target_col) == 0)
                & (pl.col("_position") >= min_history)
                & (pl.col("_rolling_mean_pre") > 0)
                & (pl.col("_rolling_mean_pre") >= threshold_ratio * pl.col("_series_mean"))
            ).alias("is_suspicious_zero")
        )
        .drop(["_rolling_mean_pre", "_series_mean", "_position"])
    )


def correct_stockouts(
    df: pl.DataFrame,
    strategy: CorrectionStrategy = "rolling_mean",
    target_col: str = "sales",
    id_col: str = "id",
    date_col: str = "date",
    rolling_window: int = 4,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Replace suspicious zeros with imputed values.

    Args:
        df: Long-format DataFrame.
        strategy: "none" leaves the data untouched; "rolling_mean" replaces
            with rolling mean; "median" replaces with non-zero series median.
        target_col, id_col, date_col: Column names.
        rolling_window: Window for the rolling mean (only used if
            strategy == "rolling_mean").

    Returns:
        Tuple of:
        - Corrected DataFrame (same shape; the target column is overwritten
          on suspicious-zero rows, the rest is unchanged).
        - Stats dict with: n_total_rows, n_suspicious, share_suspicious,
          mean_imputed_value.
    """
    if strategy == "none":
        return df, {
            "n_total_rows": df.height,
            "n_suspicious": 0,
            "share_suspicious": 0.0,
            "mean_imputed_value": float("nan"),
        }

    flagged = detect_suspicious_zeros(
        df,
        target_col=target_col,
        id_col=id_col,
        date_col=date_col,
        rolling_window=rolling_window,
    )
    n_suspicious = int(flagged["is_suspicious_zero"].sum())

    if n_suspicious == 0:
        # Nothing to correct
        return df, {
            "n_total_rows": df.height,
            "n_suspicious": 0,
            "share_suspicious": 0.0,
            "mean_imputed_value": float("nan"),
        }

    if strategy == "rolling_mean":
        # Compute the imputation: rolling mean of the previous N periods
        with_imputation = flagged.with_columns(
            pl.col(target_col)
            .rolling_mean(window_size=rolling_window, min_samples=2)
            .shift(1)
            .over(id_col)
            .alias("_imputed")
        )
    elif strategy == "median":
        # Use non-zero median of the series
        with_imputation = flagged.with_columns(
            pl.when(pl.col(target_col) > 0)
            .then(pl.col(target_col))
            .otherwise(None)
            .median()
            .over(id_col)
            .alias("_imputed")
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    corrected = with_imputation.with_columns(
        pl.when(pl.col("is_suspicious_zero"))
        .then(pl.col("_imputed"))
        .otherwise(pl.col(target_col))
        .fill_null(pl.col(target_col))  # if imputed is null, keep original
        .alias(target_col)
    ).drop(["is_suspicious_zero", "_imputed"])

    mean_imputed = (
        flagged.filter(pl.col("is_suspicious_zero"))
        .select(
            pl.col(target_col)
            .rolling_mean(window_size=rolling_window, min_samples=2)
            .shift(1)
            .over(id_col)
        )
        .to_series()
        .mean()
    )

    return corrected, {
        "n_total_rows": df.height,
        "n_suspicious": n_suspicious,
        "share_suspicious": n_suspicious / df.height,
        "mean_imputed_value": float(mean_imputed) if mean_imputed is not None else float("nan"),
    }
