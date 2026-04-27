# Data

This directory holds all data artifacts. **No raw or processed data is committed to git** — see `.gitignore`.

## Structure

- `raw/` — Original M5 dataset, untouched
- `interim/` — Cleaned intermediate data
- `processed/` — Final feature sets ready for training

## Downloading M5

The M5 Forecasting dataset is downloaded from Kaggle:

```bash
# 1. Install Kaggle CLI (already in dev dependencies)
# 2. Place your kaggle.json in ~/.kaggle/ (chmod 600)
# 3. Run:
make data
```

This will populate `data/raw/` with:
- `calendar.csv` — Date-level calendar features (day of week, events, SNAP)
- `sales_train_evaluation.csv` — Daily unit sales for ~30k SKUs × 10 stores
- `sell_prices.csv` — Weekly prices per SKU × store
- `sample_submission.csv` — Submission format reference

## Why no data in git?

- Files are large (~500 MB raw)
- Reproducibility comes from the *download script* and exact data version, not committed files
- Allows the repo to scale to other datasets without bloat
