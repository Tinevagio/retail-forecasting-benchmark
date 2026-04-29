# %% [markdown]
# # 01 — EDA Overview: M5 Forecasting Dataset
#
# **Goal of this notebook**: build a structured, business-aware understanding
# of the M5 dataset that informs every modeling choice in subsequent phases.
#
# **Guiding questions** (each section below answers one or more):
# 1. What is the structure, granularity, and time range of the data?
# 2. What seasonal and calendar patterns are exploitable?
# 3. What pitfalls should we anticipate (stockouts, missing data, regime changes)?
# 4. How heterogeneous is the demand (rotation, intermittence)?
# 5. Where will Holt-Winters likely fail and ML likely win?
#
# Each finding is summarized as a **takeaway** at the end of its section.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from forecasting.data.load import load_calendar, load_prices, load_sales, melt_sales

sns.set_theme(style="whitegrid")
pl.Config.set_tbl_rows(20)

# %% [markdown]
# ## 1. Load the data and inspect structure
#
# **Question**: what are the dimensions of the problem? How many SKUs, stores,
# states, days? What's the time span?

# %%
calendar = load_calendar()
sales_wide = load_sales()
prices = load_prices()

print(f"Calendar:    {calendar.shape}  -> {calendar['date'].min()} to {calendar['date'].max()}")
print(f"Sales (wide):{sales_wide.shape}")
print(f"Prices:      {prices.shape}")

# %%
# Sneak peek at each table
print("=" * 80)
print("CALENDAR")
print("=" * 80)
print(calendar.head())

print("\n" + "=" * 80)
print("SALES (wide format - first 6 columns + 3 day columns)")
print("=" * 80)
day_cols = [c for c in sales_wide.columns if c.startswith("d_")]
print(sales_wide.select(sales_wide.columns[:6] + day_cols[:3]).head())

print("\n" + "=" * 80)
print("PRICES")
print("=" * 80)
print(prices.head())

# %%
# Hierarchy summary
print("\nHIERARCHY:")
print(f"  States:      {sales_wide['state_id'].n_unique()}")
print(f"  Stores:      {sales_wide['store_id'].n_unique()}")
print(
    f"  Categories:  {sales_wide['cat_id'].n_unique()}  -> {sales_wide['cat_id'].unique().to_list()}"
)
print(
    f"  Departments: {sales_wide['dept_id'].n_unique()}  -> {sales_wide['dept_id'].unique().to_list()}"
)
print(f"  Items:       {sales_wide['item_id'].n_unique()}")
print(f"  SKU x Store: {sales_wide.shape[0]:,} series")
print(f"  Days:        {len(day_cols):,}")

# %% [markdown]
# **Takeaway 1 (observed)**:
# We have a rich hierarchy — 3 categories → 7 departments → ~3,000 items → ~30k
# SKUxstore series over ~5.4 years (1,941 days). This hierarchy will be central
# for both feature engineering (transfer information across siblings) and
# reconciliation (phase 3).

# %% [markdown]
# ## 2. Convert to long format for analysis
#
# Wide format is space-efficient on disk but unworkable for analysis.
# We melt to one row per (SKU, store, day).

# %%
sales = melt_sales(sales_wide, calendar=calendar)
print(f"Long format: {sales.shape}")
print(sales.head())
print(f"\nMemory usage: {sales.estimated_size('mb'):.1f} MB")

# %% [markdown]
# ## 3. Aggregate seasonality patterns
#
# **Question**: what seasonal patterns exist at the aggregated level?
# Multiple time scales matter: weekly, monthly, yearly.

# %%
# Daily total sales across the entire portfolio
daily_total = sales.group_by("date").agg(pl.col("sales").sum().alias("total_sales")).sort("date")

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily_total["date"], daily_total["total_sales"], linewidth=0.5)
ax.set_title("Total daily sales — full portfolio")
ax.set_ylabel("Units sold")
plt.tight_layout()
plt.show()

# %% [markdown]
# Look for: long-term trend, annual seasonality bumps, weekly oscillation,
# and any breaks/regime changes. Christmas dips are notable in M5 because
# Walmart stores close on Christmas day.

# %%
# Day-of-week pattern
sales_with_dow = sales.join(calendar.select(["d", "weekday", "wday"]), on="d", how="left")
dow_avg = (
    sales_with_dow.group_by(["weekday", "wday"])
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .sort("wday")
)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(dow_avg["weekday"], dow_avg["avg_sales"])
ax.set_title("Average sales by day of week")
ax.set_ylabel("Mean units / SKU / day")
plt.tight_layout()
plt.show()

# %%
# Monthly seasonality (annual pattern)
monthly = (
    sales.with_columns(pl.col("date").dt.month().alias("month"))
    .group_by("month")
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .sort("month")
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(monthly["month"], monthly["avg_sales"])
ax.set_xticks(range(1, 13))
ax.set_title("Average sales by month of year")
ax.set_ylabel("Mean units / SKU / day")
plt.tight_layout()
plt.show()

# %%
# Quantify the weekly amplitude
dow_ratio = dow_avg["avg_sales"].max() / dow_avg["avg_sales"].min()
print(f"Weekly amplitude (max DOW / min DOW): {dow_ratio:.2f}x")
print(f"Peak DOW: {dow_avg.sort('avg_sales', descending=True)['weekday'][0]}")
print(f"Trough DOW: {dow_avg.sort('avg_sales')['weekday'][0]}")

# %%
# Monthly seasonality BY CATEGORY (essential for differentiated takeaways)
monthly_cat = (
    sales.with_columns(pl.col("date").dt.month().alias("month"))
    .group_by(["cat_id", "month"])
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .sort(["cat_id", "month"])
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, cat in zip(axes, sorted(sales["cat_id"].unique()), strict=True):
    sub = monthly_cat.filter(pl.col("cat_id") == cat)
    ax.bar(sub["month"], sub["avg_sales"])
    ax.set_title(cat)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month")
axes[0].set_ylabel("Mean units / SKU / day")
plt.tight_layout()
plt.show()

# Quantify amplitude per category
amplitude_cat = (
    monthly_cat.group_by("cat_id")
    .agg(
        [
            pl.col("avg_sales").max().alias("max"),
            pl.col("avg_sales").min().alias("min"),
        ]
    )
    .with_columns((pl.col("max") / pl.col("min")).alias("amplitude_ratio"))
)
print(amplitude_cat)

# %%
# Christmas dip check
xmas_sales = daily_total.with_columns(
    [
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.day().alias("day"),
    ]
).filter((pl.col("month") == 12) & (pl.col("day") == 25))
print("Total sales on Dec 25th, by year:")
print(xmas_sales)


# %% [markdown]
# **Takeaway 2 (observed)**:
# - **Strong weekly seasonality**: amplitude ratio 1.38x (Saturday peak vs Wednesday trough)
#   → both Holt-Winters and ML will capture this well; not a differentiating signal
# - **Weak monthly seasonality**: amplitude ratios 1.09-1.16x across all 3 categories
#   → the monthly Holt-Winters baselines will have limited seasonal signal to exploit
# - **Christmas closure**: Walmart stores close Dec 25; total sales drop to <20 units
#   (vs ~50k+ on normal days). These 5 days require explicit handling (masking or
#   imputation) to avoid biasing seasonal estimates.
#
# **Strategic implication**: the ML edge over Holt-Winters will NOT come from
# better seasonal modeling. It will come from exogenous variables (events, SNAP,
# promos) and non-seasonal effects (cold-start, stockouts).

# %% [markdown]
# ## 4. Calendar events and SNAP days
#
# **Question**: how big are the demand spikes from events and SNAP days?
# Holt-Winters is blind to these — they are our primary ML edge.

# %%
# Fraction of days with events
event_days = calendar.filter(pl.col("event_name_1").is_not_null()).height
print(
    f"Days with at least 1 event: {event_days} / {calendar.height} ({100*event_days/calendar.height:.1f}%)"
)
print(f"\nUnique event types:\n{calendar['event_type_1'].drop_nulls().value_counts()}")

# %%
# Sales lift on event days vs non-event days (per category)
sales_with_events = sales.join(
    calendar.select(["d", "event_type_1"]), on="d", how="left"
).with_columns(pl.col("event_type_1").is_not_null().alias("is_event"))

event_lift = (
    sales_with_events.group_by(["cat_id", "is_event"])
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .pivot(values="avg_sales", index="cat_id", on="is_event")
    .with_columns(((pl.col("true") - pl.col("false")) / pl.col("false") * 100).alias("lift_pct"))
)
print("Sales lift on event days vs regular days, by category:")
print(event_lift)

# %%
# SNAP days (food benefits redemption days) - California only for illustration
snap_lift = (
    sales.filter(pl.col("state_id") == "CA")
    .join(calendar.select(["d", "snap_CA"]), on="d", how="left")
    .group_by(["cat_id", "snap_CA"])
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .pivot(values="avg_sales", index="cat_id", on="snap_CA")
)
print("\nSales on SNAP-CA days vs regular days, by category (CA stores only):")
print(snap_lift)

# %%
# Lift per specific event TYPE (not just event vs no-event)
event_type_lift = (
    sales.join(
        calendar.select(["d", "event_type_1"]),
        on="d",
        how="left",
    )
    .group_by(["cat_id", "event_type_1"])
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .sort(["cat_id", "event_type_1"])
)

# Get baseline (no event) per category
baseline = event_type_lift.filter(pl.col("event_type_1").is_null()).select(
    ["cat_id", pl.col("avg_sales").alias("baseline")]
)

event_type_lift = (
    event_type_lift.filter(pl.col("event_type_1").is_not_null())
    .join(baseline, on="cat_id", how="left")
    .with_columns(
        ((pl.col("avg_sales") - pl.col("baseline")) / pl.col("baseline") * 100).alias("lift_pct")
    )
    .sort(["cat_id", "lift_pct"], descending=[False, True])
)
print(event_type_lift)

# %%
# Investigate: which specific events are driving the negative lift?
# We'll look at the daily total sales on each event day vs normal days
event_calendar_detail = calendar.filter(pl.col("event_name_1").is_not_null()).select(
    ["d", "date", "event_name_1", "event_type_1"]
)

daily_totals = sales.group_by("date").agg(pl.col("sales").sum().alias("total_sales"))

event_day_sales = event_calendar_detail.join(daily_totals, on="date", how="left").sort(
    "total_sales"
)

print("=== 15 LOWEST-volume event days (likely store closures) ===")
print(event_day_sales.head(15))

print("\n=== 15 HIGHEST-volume event days (real demand boosters) ===")
print(event_day_sales.tail(15).sort("total_sales", descending=True))

# %% [markdown]
# **Takeaway 3 (observed)**:
# - Events affect 8.2% of days (162/1969), with 4 main types: Religious (55),
#   National (52), Cultural (37), Sporting (18).
# - **Critical finding**: aggregated "event vs no-event" lift is NEGATIVE for all
#   3 categories (-3% to -9%). This is NOT a real demand effect but an artifact
#   of mixing 3 distinct phenomena:
#   1. **Total store closures** on Christmas (~0% activity, 5 days)
#   2. **Partial closures / reduced hours** on Thanksgiving (~40% of normal activity)
#   3. **Genuine demand boosters** on specific religious dates (Pesach End, Purim
#      End) and Labor Day, with sales above 90th percentile of all days
# - Data hygiene: 4 event dates fall outside the available sales window
#   (post-d_1941); these must be filtered before any lift calculation.
# - SNAP days (CA): clean +10% lift on FOODS, smaller +3% on HOUSEHOLD/HOBBIES
#
# **Strategic implications**:
# 1. ML models must use **named-event encoding** (one feature per event), not a
#    generic "is_event" flag. Aggregating events would mean learning a near-zero
#    average effect.
# 2. Closure days (Christmas, partly Thanksgiving) require explicit handling —
#    either masking them from training or modeling them as a structural feature.
# 3. SNAP is a clean, stable, deterministic signal: this is the most reliable
#    feature opportunity. Holt-Winters cannot capture it; ML will.

# %% [markdown]
# ## 5. Intermittence and demand sparsity
#
# **Question**: how many of our 30k series are "easy" (regular demand) vs
# "intermittent" (mostly zeros)? This drastically changes which models are appropriate.

# %%
# % zeros per SKU x store
zero_share = sales.group_by("id").agg(
    [
        (pl.col("sales") == 0).mean().alias("zero_share"),
        pl.col("sales").mean().alias("avg_sales"),
        pl.col("sales").sum().alias("total_sales"),
    ]
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(zero_share["zero_share"], bins=50, edgecolor="black")
ax.set_xlabel("% of days with zero sales")
ax.set_ylabel("Number of SKU x store series")
ax.set_title("Distribution of zero-sale share across series")
plt.tight_layout()
plt.show()

print(f"\nSeries with >50% zeros: {(zero_share['zero_share'] > 0.5).sum()} / {zero_share.height}")
print(f"Series with >80% zeros: {(zero_share['zero_share'] > 0.8).sum()} / {zero_share.height}")
print(f"Series with <10% zeros: {(zero_share['zero_share'] < 0.1).sum()} / {zero_share.height}")

# %% [markdown]
# **Takeaway 4 (observed)**:
# - Only **1.2% of series are "high volume"** (<10% zeros): 356 / 30,490
# - **78% of series have >50% zeros**: 23,852 / 30,490
# - **38% of series are extremely intermittent** (>80% zeros): 11,664 / 30,490
# - The distribution is heavily right-skewed: most series live in the 70-95% zero range
#
# **Strategic implications**:
# 1. Holt-Winters' Gaussian error assumption is structurally violated on most
#    series. Standard HW will produce biased forecasts and unreliable prediction
#    intervals on the long tail.
# 2. Plain L2-regression LightGBM is also suboptimal on intermittent data.
#    Tweedie or Poisson loss are more appropriate for count data with excess zeros.
# 3. The case for hierarchical aggregation (SKU x warehouse x week vs SKU x store
#    x day) is statistically motivated, not just operationally convenient: the
#    daily granularity has unworkable sparsity, but weekly warehouse-level
#    aggregation will be much denser.
# 4. Standard MAE/RMSE are misleading on intermittent series. We will report
#    bias and MASE alongside, and segment performance analysis by intermittence
#    profile.


# %% [markdown]
# ## 6. Stockouts and the "zero ambiguity" problem
#
# **Question**: when we see a zero, is it "no demand" or "stockout"?
# This is a critical supply-chain insight that data scientists often miss.
# A zero surrounded by normal sales is suspicious. We can flag these.


# %%
# Heuristic: a "suspicious zero" is a day with sales=0 AND
# the average sales in the surrounding 14 days (excluding day) is > 1
def detect_suspicious_zeros(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["id", "date"])
        .with_columns(
            [
                pl.col("sales")
                .rolling_mean(window_size=15, center=True, min_samples=10)
                .over("id")
                .alias("rolling_mean"),
            ]
        )
        .with_columns(
            ((pl.col("sales") == 0) & (pl.col("rolling_mean") > 1)).alias("suspicious_zero")
        )
    )


# Run on a sample for speed
sample_ids = sales["id"].unique().sample(500, seed=42)
sample = sales.filter(pl.col("id").is_in(sample_ids.implode()))
sample_flagged = detect_suspicious_zeros(sample)

n_suspicious = sample_flagged["suspicious_zero"].sum()
total_zeros = (sample_flagged["sales"] == 0).sum()
print("On a 500-SKU sample:")
print(f"  Total zero-sale days:     {total_zeros}")
print(
    f"  Suspicious zeros (likely stockouts): {n_suspicious}  ({100*n_suspicious/total_zeros:.1f}%)"
)


# %%
# Distribution: how concentrated are suspicious zeros?
# Are stockouts spread evenly across SKUs or concentrated on a few?
stockout_by_sku = (
    sample_flagged.group_by("id")
    .agg(
        [
            pl.col("suspicious_zero").sum().alias("n_suspicious"),
            (pl.col("sales") == 0).sum().alias("n_zeros"),
            pl.col("sales").mean().alias("avg_sales"),
        ]
    )
    .filter(pl.col("avg_sales") > 0)  # Only series with some sales
    .with_columns(
        (pl.col("n_suspicious") / pl.col("n_zeros").cast(pl.Float64)).alias("suspicious_share")
    )
    .sort("n_suspicious", descending=True)
)

print(
    f"SKUs with at least 1 suspicious zero: "
    f"{(stockout_by_sku['n_suspicious'] > 0).sum()} / {stockout_by_sku.height}"
)
print("\nTop 10 SKUs by number of suspicious zeros:")
print(stockout_by_sku.head(10))

# %%
# Stockout intensity by category
stockout_by_cat = (
    sample_flagged.group_by("cat_id")
    .agg(
        [
            pl.col("suspicious_zero").sum().alias("n_suspicious"),
            (pl.col("sales") == 0).sum().alias("n_zeros"),
            pl.col("sales").count().alias("n_obs"),
        ]
    )
    .with_columns(
        [
            (pl.col("n_suspicious") / pl.col("n_zeros").cast(pl.Float64) * 100).alias(
                "pct_suspicious_among_zeros"
            ),
            (pl.col("n_suspicious") / pl.col("n_obs").cast(pl.Float64) * 100).alias(
                "pct_suspicious_among_all"
            ),
        ]
    )
)
print("Stockout intensity by category (sample):")
print(stockout_by_cat)

# %% [markdown]
# **Takeaway 5 (observed)**:
# - On a 500-SKU sample, **8.2% of zero-sale days are likely stockouts** (zeros
#   surrounded by normal sales activity). This is a conservative estimate;
#   the true figure is likely higher.
# - Stockouts are **concentrated** on a subset of SKUs: 75% of sampled SKUs have
#   at least one suspicious zero, and the top stockout-prone SKUs have >90% of
#   their zeros flagged as suspicious — these are likely high-rotation products
#   in chronic supply tension.
# - **FOODS suffers 2x more stockouts** than HOUSEHOLD or HOBBIES (11.7% vs ~6%
#   of zero-days flagged), consistent with shorter shelf-life and faster
#   replenishment cycles in fresh categories.
#
# **Strategic implications**:
# 1. Treating all zeros as "no demand" introduces a systematic downward bias
#    in forecasts, which is invisible in standard metrics like RMSE but causes
#    real-world stockouts to compound (self-fulfilling under-forecast spiral).
# 2. **Phase 2 will implement explicit stockout detection and imputation**, with
#    bias forecasting reported alongside accuracy metrics. This is one of the
#    project's strongest differentiating angles, drawing directly from
#    supply-chain practitioner intuition.
# 3. The current heuristic (rolling mean > 1) is a starting point, not a final
#    answer. Refinements include using `sell_prices` data to confirm whether
#    a SKU was even referenced on a given week (no price = no SKU listed),
#    which would clean up false positives.

# %% [markdown]
# ## 7. Promotions: the elephant in the room
#
# **Question**: how often are products promoted, and how big is the demand lift?
# Promotions are nowhere in the calendar — we have to detect them from price drops.

# %%
# Detect price drops: for each (item, store), flag weeks where price < 95% of trailing 8-week median
prices_with_promo = (
    prices.sort(["item_id", "store_id", "wm_yr_wk"])
    .with_columns(
        pl.col("sell_price")
        .rolling_median(window_size=8, min_samples=4)
        .over(["item_id", "store_id"])
        .alias("ref_price")
    )
    .with_columns((pl.col("sell_price") < 0.95 * pl.col("ref_price")).alias("on_promo"))
)

print(
    f"Promotion frequency: {100 * prices_with_promo['on_promo'].mean():.1f}% of (item, store, week) tuples"
)

# %%
# Sales lift on promo weeks vs non-promo weeks
# We need to join sales (daily) with prices (weekly via wm_yr_wk)
sales_with_promo = (
    sales.join(calendar.select(["d", "wm_yr_wk"]), on="d", how="left")
    .join(
        prices_with_promo.select(["item_id", "store_id", "wm_yr_wk", "on_promo"]),
        on=["item_id", "store_id", "wm_yr_wk"],
        how="left",
    )
    .with_columns(pl.col("on_promo").fill_null(False))
)

promo_lift = (
    sales_with_promo.group_by(["cat_id", "on_promo"])
    .agg(pl.col("sales").mean().alias("avg_sales"))
    .pivot(values="avg_sales", index="cat_id", on="on_promo")
    .with_columns(((pl.col("true") / pl.col("false") - 1) * 100).alias("lift_pct"))
    .sort("lift_pct", descending=True)
)
print("Sales lift on promo weeks, by category:")
print(promo_lift)

# %% [markdown]
# **Takeaway 6 (observed)**:
# - With a conservative price-drop heuristic (>5% drop vs 8-week rolling median),
#   only **1.0% of (item, store, week) tuples are flagged as promotional**. This
#   is suspiciously low for grocery retail and likely reflects:
#   1. The heuristic is too strict — many real promos are <5%
#   2. Walmart's "Every Day Low Price" model uses non-price levers (display,
#      placement, bundles) that are invisible in the dataset
# - **Sales lift on flagged promo weeks is massive**: +81% on HOBBIES, +46% on
#   FOODS — a clean signal that Holt-Winters cannot exploit.
# - HOUSEHOLD shows -1.5% lift, likely an artifact: structural price changes get
#   misidentified as promos by the rolling-median heuristic.
#
# **Strategic implications**:
# 1. Promo features will deliver large gains on the affected weeks, but the
#    overall WMAPE impact will be modest given the low frequency. The right
#    narrative is "operational impact" (avoiding stockouts on promo) rather
#    than "headline accuracy lift".
# 2. Phase 2 will refine the heuristic with category-specific thresholds and
#    direct features (price_relative_to_ref, price_drop_magnitude) rather than
#    a binary flag — both for granularity and for HOUSEHOLD-type artifacts.

# %% [markdown]
# ## 8. Cold start: new products
#
# **Question**: how many products appear mid-history? These are impossible for
# Holt-Winters to forecast (no history) but ML can leverage hierarchical
# information from sibling products.

# %%
# First sale day per SKU x store
first_sale = (
    sales.filter(pl.col("sales") > 0)
    .group_by("id")
    .agg(pl.col("date").min().alias("first_sale_date"))
)

# How many series start AFTER the dataset begins?
dataset_start = sales["date"].min()
cold_start = first_sale.filter(pl.col("first_sale_date") > dataset_start)
print(f"Total series:                       {first_sale.height}")
print(
    f"Series with cold-start (late entry): {cold_start.height}  ({100*cold_start.height/first_sale.height:.1f}%)"
)

# Distribution of cold-start dates
fig, ax = plt.subplots(figsize=(12, 4))
ax.hist(cold_start["first_sale_date"], bins=50)
ax.set_title("Distribution of first-sale dates (cold-start products)")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Takeaway 7 (observed)**:
# - **77.1% of series have a cold-start** (first sale after the dataset starts):
#   23,511 / 30,490. New products arrive at a steady rate of ~500-1000 per
#   month throughout 2011-2015.
# - True modeling cold-start (series with <28-56 days of history at forecast
#   time) is a smaller subset, but the steady arrival pattern means it's a
#   recurring production challenge, not an edge case.
#
# **Strategic implications**:
# 1. Holt-Winters cannot meaningfully forecast new products. Hierarchical
#    feature transfer (department mean, category mean, similar-SKU embeddings)
#    will be essential.
# 2. Phase 2 will treat cold-start as a first-class evaluation segment, not
#    just an aggregate metric, since model performance varies dramatically by
#    history length.


# %% [markdown]
# ## 9. Final synthesis: strategic implications for phase 2
#
# This EDA was structured to inform every modeling choice in subsequent phases.
# Below is the consolidated view of what we learned and how it shapes the work ahead.
#
# ### Summary of observations
#
# | Phenomenon | Observation | HW capability | ML opportunity |
# |---|---|---|---|
# | Weekly seasonality | Amplitude 1.38x (Sat peak vs Wed trough) | **Strong** | Marginal (already captured) |
# | Monthly seasonality | Amplitude 1.09-1.16x across categories | Adequate | Marginal |
# | Christmas closure | 5 days at <0.05% of normal volume | **Fails** | Explicit handling needed |
# | Named events | Heterogeneous: closures, partial closures, real boosters | **Fails** | Per-event encoding |
# | SNAP days (CA) | +10% on FOODS, deterministic | **Fails** | Clean win |
# | Intermittence | 78% of series with >50% zeros, 38% with >80% zeros | **Statistically violated** | Tweedie/Poisson loss, segment-aware modeling |
# | Stockouts | ~8% of zeros are likely lost sales (12% on FOODS) | **Biased downward** | Detection + imputation, bias metric |
# | Promotions | 1% of weeks (conservative), +46-81% lift when active | **Fails** | Price-derived features |
# | Cold-start | 77% of series, ~500-1000 new SKUs/month | **Cannot forecast** | Hierarchical transfer |
#
# ### Where ML will and will not win
#
# **High-confidence ML wins** (HW structurally cannot capture):
# - SNAP-day forecasts on FOODS
# - Promotional weeks on HOBBIES and FOODS
# - Cold-start products (no historical signal)
# - Days following stockouts (HW learns biased patterns)
#
# **Likely ML wins** (HW can do something but is suboptimal):
# - Intermittent series (better loss function)
# - Named events with heterogeneous effects (per-event encoding)
#
# **Marginal differences expected**:
# - High-volume, stable, non-promoted SKUs (HW captures the dominant weekly signal)
#
# ### Phase 2 priorities, in order
#
# 1. **Build well-tuned Holt-Winters baseline** (per-track: weekly fresh, monthly
#    dry/non-food). A weak baseline would inflate ML gains and undermine credibility.
# 2. **Calendar feature engineering**: per-event encoding (not aggregate flags),
#    SNAP days, day-of-week x month interactions, closure handling.
# 3. **Promo features**: price-relative-to-reference, drop magnitude, weeks
#    since last promo — beyond a binary flag.
# 4. **Hierarchical features**: department mean, category mean, store mean — the
#    main lever for cold-start and low-volume SKUs.
# 5. **Stockout detection and imputation**: implement detection (refined heuristic
#    using prices), then evaluate impact on bias forecasting specifically.
# 6. **Loss function choice**: Tweedie or Poisson on intermittent segments,
#    standard L2 on volume segments. Possibly two-stage models on extreme tail.
# 7. **Segmented evaluation**: report metrics by intermittence tier, by category,
#    by promotion status, by cold-start status — not just headline numbers.
#
# ### A non-trivial methodological commitment
#
# The most differentiating aspect of this project is the explicit recognition
# that ML does not win uniformly. Section 9 above lists the segments where ML
# is structurally advantaged and those where Holt-Winters remains competitive.
# Phase 4 will conclude with a **routing strategy recommendation**: which model
# to use for which segment, rather than a blanket replacement claim.
