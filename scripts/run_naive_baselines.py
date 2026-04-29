"""Quick-check: run naive baselines on the fresh_weekly track."""

from forecasting.data.aggregate import TRACK_CONFIGS, prepare_track
from forecasting.data.load import load_calendar, load_sales, melt_sales
from forecasting.data.splits import make_walk_forward_folds
from forecasting.evaluation.runner import aggregate_by_fold, evaluate_models
from forecasting.models.naive import DriftNaive, HistoricalMean, SeasonalNaive

# 1. Load and prepare the fresh_weekly track
calendar = load_calendar()
sales_wide = load_sales()
sales = melt_sales(sales_wide, calendar=calendar)

# Need cat_id and dept_id columns to filter; melt_sales already includes them
track_data = prepare_track(sales, "fresh_weekly")
print(f"Fresh weekly track: {track_data.shape}")

# 2. Walk-forward folds: 4 folds of 4 weeks each (28 days)
cfg = TRACK_CONFIGS["fresh_weekly"]
horizon_weeks = cfg["horizon_periods"]
folds = make_walk_forward_folds(
    min_date=track_data["date"].min(),
    max_date=track_data["date"].max(),
    n_folds=4,
    test_horizon_days=horizon_weeks * 7,
)
for f in folds:
    print(f)

# 3. Evaluate naive baselines
models = [
    HistoricalMean(frequency="W"),
    SeasonalNaive(season_length=cfg["seasonality"], frequency="W"),
    DriftNaive(frequency="W"),
]
results = evaluate_models(models, track_data, folds, horizon=horizon_weeks)
summary = aggregate_by_fold(results)
print("\n=== Summary (sorted by WMAPE) ===")
print(summary)
