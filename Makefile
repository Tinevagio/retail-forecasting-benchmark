# Makefile — common development and deployment workflows
#
# A Makefile here is mostly documentation: it lists the verbs that matter
# in the project and gives them short, memorable names. New contributors
# (and future-you) get a one-line cheatsheet via `make help`.
#
# Tested on Linux, macOS, and Git Bash on Windows. Some targets require
# Docker Desktop running.

.PHONY: help install test lint format check train serve docker-build docker-run docker-stop docker-logs clean

# Default: print help when running `make` with no args
.DEFAULT_GOAL := help

# ---- Help -----------------------------------------------------------------

help:  ## Show this help message
	@echo "Usage: make <target>"
	@echo ""
	@echo "Local development:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---- Local dev ------------------------------------------------------------

install:  ## Install dependencies (creates/updates the venv)
	uv sync --all-extras

test:  ## Run the test suite (excluding slow tests)
	uv run pytest -v -m "not slow"

test-all:  ## Run all tests including slow integration tests
	uv run pytest -v

lint:  ## Run ruff and mypy
	uv run ruff check src/ tests/ scripts/
	uv run mypy src/

format:  ## Format the code with ruff
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

check: format lint test  ## Run format + lint + test (pre-commit safety net)

# ---- Training and serving (local Python, no Docker) ----------------------

train:  ## Train the LightGBM model on store CA_1 (~3 min)
	uv run python scripts/train_for_serving.py --sample-stores CA_1

serve:  ## Run the API in development mode (auto-reload on code changes)
	uv run uvicorn forecasting.serving.api:app --reload

# ---- Docker workflows -----------------------------------------------------

docker-build:  ## Build the API Docker image
	docker compose build

docker-run:  ## Start the API in Docker (detached)
	docker compose up -d

docker-stop:  ## Stop the running container
	docker compose down

docker-logs:  ## Follow the running container's logs
	docker compose logs -f api

docker-shell:  ## Open a shell in the running container (debugging)
	docker compose exec api /bin/bash || docker compose exec api /bin/sh

# ---- Cleanup --------------------------------------------------------------

clean:  ## Remove caches, build artifacts (keeps trained models)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} \;
	find . -type d -name .ipynb_checkpoints -prune -exec rm -rf {} \;
