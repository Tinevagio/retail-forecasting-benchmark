.PHONY: help install install-dev data clean test lint format check baseline train-lgbm evaluate

help:  ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	uv sync

install-dev:  ## Install dev dependencies and pre-commit hooks
	uv sync --all-extras
	uv run pre-commit install

data:  ## Download M5 dataset from Kaggle
	uv run python scripts/download_data.py

clean:  ## Remove cache and temporary files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name *.egg-info -exec rm -rf {} +
	rm -rf htmlcov/ .coverage

test:  ## Run unit tests with coverage
	uv run pytest

test-fast:  ## Run only fast tests
	uv run pytest -m "not slow"

lint:  ## Run linters
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/

format:  ## Format code
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

check: lint test  ## Run all checks (lint + test)

baseline:  ## Train Holt-Winters baseline (all tracks)
	uv run python scripts/train_baseline.py

train-lgbm:  ## Train LightGBM model
	uv run python scripts/train_ml.py --model lgbm

evaluate:  ## Run full evaluation and generate reports
	uv run python scripts/evaluate.py

mlflow-ui:  ## Launch MLflow UI on port 5000
	uv run mlflow ui --port 5000
