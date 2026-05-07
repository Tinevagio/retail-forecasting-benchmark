"""Stratified sampling of SKU x store series for tractable training.

Why this exists
---------------
The full M5 daily SKU x store grid is ~30k series x ~1900 days = ~57M rows.
That is too much for slow per-series models like HoltWinters AutoETS, which
trains one ETS per series and takes ~2h on the 8230 series of fresh_weekly.

For the project 2 stock optimization use case we keep the daily grain (lead
times of a few days are not representable at weekly grain) but downscale to
a stratified sample. 1000 series x 1900 days = 1.9M rows is comfortable.

Stratification
--------------
We stratify on `(cat_id, state_id)` by default. With 3 categories (FOODS,
HOUSEHOLD, HOBBIES) and 3 states (CA, TX, WI), that's 9 strata. Each strata
contributes a number of series proportional to its share of the population,
so the sample inherits the natural imbalance (FOODS dominates) without being
biased toward any particular cell.

Determinism
-----------
A seed makes the sample reproducible across runs, machines and re-installs.
The classic polars trap (Series.unique() does not preserve order) is handled
explicitly: we sort by `id` before sampling, so the same seed always picks
the same series — a lesson from project 1, where this was a real bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


def _allocate_targets(
    counts: pl.DataFrame,
    strata: tuple[str, ...],
    total_series: int,
    n_skus: int,
) -> dict[tuple[object, ...], int]:
    """Proportional allocation across strata with rounding reconciliation."""
    target_alloc: dict[tuple[object, ...], int] = {}
    for row in counts.iter_rows(named=True):
        strata_key = tuple(row[c] for c in strata)
        share = row["n_in_strata"] / total_series
        target_alloc[strata_key] = max(1, round(n_skus * share))

    total_target = sum(target_alloc.values())
    if total_target != n_skus:
        # Absorb the rounding error in the largest strata
        largest = max(target_alloc, key=lambda k: target_alloc[k])
        target_alloc[largest] += n_skus - total_target
    return target_alloc


def _reconcile_with_actual_sizes(
    target_alloc: dict[tuple[object, ...], int],
    actual_sizes: dict[tuple[object, ...], int],
) -> dict[tuple[object, ...], int]:
    """Cap each strata to its actual size; reallocate any deficit to others."""
    deficit = 0
    for strata_key, target in target_alloc.items():
        if target > actual_sizes[strata_key]:
            deficit += target - actual_sizes[strata_key]
            target_alloc[strata_key] = actual_sizes[strata_key]

    if deficit > 0:
        for strata_key in sorted(
            target_alloc, key=lambda k: actual_sizes[k] - target_alloc[k], reverse=True
        ):
            room = actual_sizes[strata_key] - target_alloc[strata_key]
            take = min(room, deficit)
            target_alloc[strata_key] += take
            deficit -= take
            if deficit == 0:
                break
    return target_alloc


def stratified_sample_skus(
    df: pl.DataFrame,
    n_skus: int,
    strata: Sequence[str] = ("cat_id", "state_id"),
    seed: int = 42,
    item_col: str = "item_id",
    store_col: str = "store_id",
) -> list[str]:
    """Return a deterministic stratified sample of SKU x store series.

    The output list contains composite ids of the form ``f"{item_id}__{store_id}"``,
    same convention as elsewhere in the project. The sample is allocated
    proportionally across strata, with the smallest-strata rounding handled
    by giving each strata at least 1 series and adjusting the largest strata
    down to keep the total at n_skus.

    Parameters
    ----------
    df : pl.DataFrame
        Long-format dataframe with at least item_col, store_col, and the
        strata columns. Each (item_id, store_id) pair is a candidate series.
        Multiple rows per series are fine (they are deduplicated here).
    n_skus : int
        Total number of series to sample. Must be > 0 and <= total number
        of unique series in df.
    strata : sequence of str, optional
        Columns defining the strata. Default ("cat_id", "state_id").
    seed : int, optional
        Seed for the per-strata sampling. Default 42.
    item_col, store_col : str, optional
        Column names for the SKU and store identifiers.

    Returns
    -------
    list[str]
        Sorted list of composite ids of the sampled series, length n_skus
        (or close to it; see edge cases below). Sorting makes the output
        order itself deterministic, useful for downstream `is_in` filters.

    Raises
    ------
    ValueError
        If n_skus <= 0, or if n_skus exceeds the total number of unique
        series, or if any strata column is missing from df.

    Edge cases
    ----------
    - If a strata has fewer series than its target allocation, we take all
      of them and reallocate the deficit to the largest strata. Total may
      end up slightly below n_skus if many strata are smaller than their
      targets, but this is rare with M5-scale data.
    - If a strata is empty, it is silently dropped (it would not contribute
      anyway).
    """
    if n_skus <= 0:
        raise ValueError(f"n_skus must be > 0, got {n_skus}")

    missing = [col for col in (*strata, item_col, store_col) if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for stratified sampling: {missing}. "
            f"Available: {sorted(df.columns)}"
        )

    # Build the unique series table with their strata.
    composite = pl.concat_str([item_col, store_col], separator="__").alias("id")
    series_df = (
        df.select(composite, *strata)
        .unique()
        .sort("id")  # IMPORTANT: explicit sort for deterministic sampling.
    )

    total_series = series_df.height
    if n_skus > total_series:
        raise ValueError(
            f"n_skus ({n_skus}) exceeds total unique series ({total_series}). "
            f"Lower n_skus or relax the upstream filters."
        )

    # Count series per strata
    counts = series_df.group_by(list(strata)).agg(pl.len().alias("n_in_strata")).sort(list(strata))
    strata_tuple = tuple(strata)

    # Allocate proportionally then reconcile against actual strata sizes
    target_alloc = _allocate_targets(counts, strata_tuple, total_series, n_skus)
    actual_sizes = {
        tuple(row[c] for c in strata): int(row["n_in_strata"])
        for row in counts.iter_rows(named=True)
    }
    target_alloc = _reconcile_with_actual_sizes(target_alloc, actual_sizes)

    # Sample within each strata
    sampled_ids: list[str] = []
    for row in counts.iter_rows(named=True):
        strata_key = tuple(row[c] for c in strata)
        target = target_alloc.get(strata_key, 0)
        if target == 0:
            continue

        strata_filter = pl.lit(value=True)
        for col, val in zip(strata, strata_key, strict=True):
            strata_filter = strata_filter & (pl.col(col) == val)

        strata_series = series_df.filter(strata_filter).sort("id")
        sample = strata_series.sample(n=target, seed=seed, with_replacement=False)
        sampled_ids.extend(sample["id"].to_list())

    sampled_ids.sort()
    return sampled_ids
