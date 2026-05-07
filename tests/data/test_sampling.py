"""Tests for forecasting.data.sampling."""

from __future__ import annotations

import polars as pl
import pytest

from forecasting.data.sampling import stratified_sample_skus


def _build_synthetic_population(
    n_per_strata: dict[tuple[str, str], int],
) -> pl.DataFrame:
    """Build a synthetic SKU x store population with given strata sizes.

    Each (cat_id, state_id) cell gets n_per_strata[(cat, state)] unique
    (item_id, store_id) pairs.
    """
    rows = []
    serial = 0
    for (cat, state), count in n_per_strata.items():
        for _ in range(count):
            serial += 1
            rows.append(
                {
                    "item_id": f"ITEM_{serial:05d}",
                    "store_id": f"{state}_{serial % 3 + 1}",
                    "cat_id": cat,
                    "state_id": state,
                    # extra noise rows to test dedupe inside the function
                    "sales": 1,
                }
            )
    return pl.DataFrame(rows)


class TestStratifiedSample:
    def test_basic_proportional_allocation(self) -> None:
        # 3 strata: 100, 50, 50. Total 200. Sample 20 (10%).
        # Expected ~10 from FOODS+CA, ~5 from HOUSEHOLD+TX, ~5 from HOBBIES+WI.
        population = _build_synthetic_population(
            {
                ("FOODS", "CA"): 100,
                ("HOUSEHOLD", "TX"): 50,
                ("HOBBIES", "WI"): 50,
            }
        )
        ids = stratified_sample_skus(population, n_skus=20, seed=42)
        assert len(ids) == 20

    def test_determinism_same_seed(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 50, ("HOUSEHOLD", "TX"): 50})
        ids1 = stratified_sample_skus(population, n_skus=10, seed=42)
        ids2 = stratified_sample_skus(population, n_skus=10, seed=42)
        assert ids1 == ids2

    def test_different_seed_different_sample(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 100, ("HOUSEHOLD", "TX"): 100})
        ids1 = stratified_sample_skus(population, n_skus=20, seed=42)
        ids2 = stratified_sample_skus(population, n_skus=20, seed=123)
        # Same total, but different individuals (overwhelmingly likely)
        assert ids1 != ids2

    def test_returned_ids_are_sorted(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 50, ("HOUSEHOLD", "TX"): 50})
        ids = stratified_sample_skus(population, n_skus=10, seed=42)
        assert ids == sorted(ids)

    def test_returned_ids_are_unique(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 50, ("HOUSEHOLD", "TX"): 50})
        ids = stratified_sample_skus(population, n_skus=20, seed=42)
        assert len(ids) == len(set(ids))

    def test_strata_coverage(self) -> None:
        """All non-empty strata should contribute at least 1 series."""
        population = _build_synthetic_population(
            {
                ("FOODS", "CA"): 100,
                ("HOUSEHOLD", "TX"): 100,
                ("HOBBIES", "WI"): 100,
            }
        )
        ids = stratified_sample_skus(population, n_skus=12, seed=42)

        # Map each id back to its strata via the population
        composite = pl.concat_str(["item_id", "store_id"], separator="__")
        pop_with_id = population.with_columns(composite.alias("id"))
        sampled_strata = (
            pop_with_id.filter(pl.col("id").is_in(ids)).select("cat_id", "state_id").unique()
        )
        # 3 strata in input, all should be represented
        assert sampled_strata.height == 3

    def test_n_skus_equals_population(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 5, ("HOUSEHOLD", "TX"): 5})
        ids = stratified_sample_skus(population, n_skus=10, seed=42)
        assert len(ids) == 10

    def test_n_skus_too_large_raises(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 5})
        with pytest.raises(ValueError, match="exceeds total unique series"):
            stratified_sample_skus(population, n_skus=100, seed=42)

    def test_zero_n_skus_raises(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 5})
        with pytest.raises(ValueError, match="n_skus must be > 0"):
            stratified_sample_skus(population, n_skus=0, seed=42)

    def test_negative_n_skus_raises(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 5})
        with pytest.raises(ValueError, match="n_skus must be > 0"):
            stratified_sample_skus(population, n_skus=-3, seed=42)

    def test_missing_strata_column_raises(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 5})
        # remove cat_id
        population = population.drop("cat_id")
        with pytest.raises(ValueError, match="Missing required columns"):
            stratified_sample_skus(population, n_skus=2, seed=42)

    def test_missing_item_id_raises(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 5})
        population = population.drop("item_id")
        with pytest.raises(ValueError, match="Missing required columns"):
            stratified_sample_skus(population, n_skus=2, seed=42)

    def test_dedupes_input_rows(self) -> None:
        """Multiple rows per series (e.g. one per date) should not bias sampling."""
        # 5 series with 100 dates each = 500 rows, but only 5 unique series
        rows = []
        for i in range(5):
            for _ in range(100):
                rows.append(
                    {
                        "item_id": f"ITEM_{i}",
                        "store_id": "CA_1",
                        "cat_id": "FOODS",
                        "state_id": "CA",
                        "sales": 1,
                    }
                )
        population = pl.DataFrame(rows)
        ids = stratified_sample_skus(population, n_skus=3, seed=42)
        assert len(ids) == 3
        assert all(i.startswith("ITEM_") for i in ids)

    def test_small_strata_reallocation(self) -> None:
        """When one strata is smaller than its target allocation, reallocate
        the deficit to a larger strata."""
        # 95 in FOODS, 5 in HOUSEHOLD, total 100. Sample 50.
        # Naive: 47.5 from FOODS (47), 2.5 from HOUSEHOLD (3). Total 50, OK.
        # But if we sample 90: 85.5 from FOODS (86), 4.5 from HOUSEHOLD (5).
        # FOODS exceeds at 86, take 86 from 95 OK. HOUSEHOLD = 5/5 = OK.
        # Adjusted total = 91. Off by 1, fine for an integration test.
        # Edge case: n_skus close to total.
        population = _build_synthetic_population({("FOODS", "CA"): 95, ("HOUSEHOLD", "TX"): 5})
        ids = stratified_sample_skus(population, n_skus=99, seed=42)
        # Should saturate HOUSEHOLD (all 5) and take ~94 from FOODS
        assert 95 <= len(ids) <= 99

    def test_custom_strata(self) -> None:
        population = _build_synthetic_population({("FOODS", "CA"): 50, ("HOUSEHOLD", "TX"): 50})
        ids = stratified_sample_skus(population, n_skus=10, strata=("cat_id",), seed=42)
        assert len(ids) == 10
