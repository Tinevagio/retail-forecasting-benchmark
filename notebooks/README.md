# Notebooks

Notebooks here are **exploratory artifacts**, not production code. The convention:

- **Numbered prefix** (`01_`, `02_`, ...) for chronological order
- **Descriptive name** in snake_case
- **Outputs stripped before commit** (handled by pre-commit `nbstripout`)
- **No long-running code** — heavy compute belongs in `scripts/` or `src/`

Notebooks should *describe findings*, not *contain implementation*. Once a piece of analysis is consolidated, refactor it into the `forecasting` package and write a test for it.

## Planned notebooks

- `01_eda_overview.ipynb` — High-level overview of M5 dataset, sanity checks
- `02_seasonality_analysis.ipynb` — Multi-scale seasonality patterns
- `03_promo_impact.ipynb` — Quantifying promotional effects, motivating ML features
- `04_intermittence.ipynb` — Long-tail and intermittent demand patterns
- `05_baseline_diagnostics.ipynb` — Where Holt-Winters wins and fails
