"""Central configuration: paths, constants, and project-wide settings."""

from pathlib import Path
from typing import Final

# ----- Paths -----
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
INTERIM_DATA_DIR: Final[Path] = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"
CONFIGS_DIR: Final[Path] = PROJECT_ROOT / "configs"

# ----- M5 specifics -----
M5_RAW_FILES: Final[dict[str, str]] = {
    "calendar": "calendar.csv",
    "sales": "sales_train_evaluation.csv",
    "prices": "sell_prices.csv",
    "submission_template": "sample_submission.csv",
}

# ----- Track definitions (mapping M5 → real-world retail tracks) -----
TRACKS: Final[dict[str, dict]] = {
    "fresh": {
        "name": "Fresh products (weekly)",
        "frequency": "W",
        "horizons": [1, 2, 3, 4],  # weeks ahead
        "m5_categories": ["FOODS_3"],  # Fast-moving foods (proxy for fresh)
        "seasonality_period": 52,
    },
    "dry": {
        "name": "Dry / grocery (monthly)",
        "frequency": "M",
        "horizons": [1, 2, 3],  # months ahead
        "m5_categories": ["FOODS_1", "FOODS_2", "HOUSEHOLD_1", "HOUSEHOLD_2"],
        "seasonality_period": 12,
    },
    "non_food": {
        "name": "Non-food (monthly)",
        "frequency": "M",
        "horizons": [1, 2, 3],
        "m5_categories": ["HOBBIES_1", "HOBBIES_2"],
        "seasonality_period": 12,
    },
}

# ----- Validation strategy -----
VALIDATION_FOLDS: Final[int] = 4
VALIDATION_GAP_DAYS: Final[int] = 0  # Gap between train and val to prevent leakage

# ----- Random seed for reproducibility -----
RANDOM_SEED: Final[int] = 42
