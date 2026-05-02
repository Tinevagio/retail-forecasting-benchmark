"""FastAPI prediction service.

Exposes endpoints to serve forecasts from pre-trained models stored on disk.
Models are loaded once at startup and kept in memory; no retraining happens
in the request path.

Endpoints:
- GET  /health                          : basic liveness check
- GET  /info                            : metadata about loaded models
- POST /predict                         : forecast for a single series
- POST /predict-batch                   : forecast for several series

Configuration:
The service reads two environment variables at startup:
- FORECASTING_ARTIFACTS_DIR : directory containing model artifacts and
                              feature stores (default: ./artifacts)

Each artifact is loaded as `<name>.pkl` with a companion feature store
`<name>_features.parquet`.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecasting.serving.feature_store import (
    get_features_for_ids,
    load_feature_store,
)
from forecasting.serving.persistence import ModelArtifact, load_artifact

# ---- Configuration --------------------------------------------------------

ARTIFACTS_DIR = Path(os.getenv("FORECASTING_ARTIFACTS_DIR", "artifacts"))


# ---- Schemas (Pydantic) ---------------------------------------------------


class PredictRequest(BaseModel):
    """Request schema for /predict."""

    series_id: str = Field(
        ...,
        description="Series identifier, e.g. 'FOODS_3_001__CA_1'.",
        examples=["FOODS_3_001__CA_1"],
    )
    horizon: int = Field(
        4,
        ge=1,
        le=12,
        description="Number of future periods to forecast.",
    )
    model_name: str = Field(
        "lightgbm-tweedie",
        description="Name of the model to use (must be available in artifacts).",
    )


class PredictBatchRequest(BaseModel):
    """Request schema for /predict-batch."""

    series_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of series identifiers to forecast.",
    )
    horizon: int = Field(4, ge=1, le=12)
    model_name: str = Field("lightgbm-tweedie")


class PredictionPoint(BaseModel):
    """A single (date, value) forecast point."""

    date: date
    prediction: float


class PredictResponse(BaseModel):
    """Response schema for /predict."""

    series_id: str
    model_name: str
    train_end: date
    predictions: list[PredictionPoint]


class PredictBatchResponse(BaseModel):
    """Response schema for /predict-batch."""

    model_name: str
    train_end: date
    series: list[PredictResponse]


class ModelInfo(BaseModel):
    """Metadata for a loaded model."""

    model_name: str
    track_name: str
    train_end: date
    feature_count: int
    trained_at: str  # ISO format


class HealthResponse(BaseModel):
    status: str
    n_models_loaded: int


# ---- Application factory --------------------------------------------------


class ModelRegistry:
    """In-memory registry of (artifact, feature_store) pairs by model name.

    Loaded once at startup. Single source of truth for the prediction
    endpoints.
    """

    def __init__(self) -> None:
        self._models: dict[str, tuple[ModelArtifact, pl.DataFrame]] = {}

    def load_from_dir(self, artifacts_dir: Path) -> None:
        """Scan the directory for *.pkl + *_features.parquet pairs.

        Filename convention: artifact `foo.pkl` is paired with feature
        store `foo_features.parquet` if the latter exists.
        """
        if not artifacts_dir.exists():
            # Empty registry is fine; the API will return clear errors
            return

        for artifact_path in artifacts_dir.glob("*.pkl"):
            try:
                artifact = load_artifact(artifact_path)
            except (ValueError, OSError):
                # Bad artifact — skip but keep loading others
                continue

            features_path = artifact_path.with_name(f"{artifact_path.stem}_features.parquet")
            if not features_path.exists():
                continue

            try:
                feature_store = load_feature_store(features_path)
            except (ValueError, OSError):
                continue

            self._models[artifact.model_name] = (artifact, feature_store)

    def get(self, model_name: str) -> tuple[ModelArtifact, pl.DataFrame]:
        if model_name not in self._models:
            raise KeyError(f"Model {model_name!r} not loaded")
        return self._models[model_name]

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    def info(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                model_name=art.model_name,
                track_name=art.track_name,
                train_end=art.train_end,
                feature_count=len(art.feature_cols),
                trained_at=art.trained_at.isoformat(),
            )
            for art, _ in self._models.values()
        ]


def create_app() -> FastAPI:
    """Build the FastAPI application and load models at startup."""
    app = FastAPI(
        title="Retail Forecasting API",
        description=(
            "Serves forecasts from pre-trained Holt-Winters and LightGBM "
            "models. See /info for available models."
        ),
        version="0.1.0",
    )

    registry = ModelRegistry()
    registry.load_from_dir(ARTIFACTS_DIR)
    app.state.registry = registry

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            n_models_loaded=len(registry.list_models()),
        )

    @app.get("/info", response_model=list[ModelInfo])
    def info() -> list[ModelInfo]:
        return registry.info()

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        try:
            artifact, feature_store = registry.get(req.model_name)
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail=(f"{e}. Available models: {registry.list_models()}"),
            ) from e

        features = get_features_for_ids(feature_store, [req.series_id])
        if features.height == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Series {req.series_id!r} not found in feature store",
            )

        predictions_df = artifact.model.predict(horizon=req.horizon, ids=[req.series_id])

        points = [
            PredictionPoint(date=row["date"], prediction=row["prediction"])
            for row in predictions_df.iter_rows(named=True)
        ]

        return PredictResponse(
            series_id=req.series_id,
            model_name=artifact.model_name,
            train_end=artifact.train_end,
            predictions=points,
        )

    @app.post("/predict-batch", response_model=PredictBatchResponse)
    def predict_batch(req: PredictBatchRequest) -> PredictBatchResponse:
        try:
            artifact, feature_store = registry.get(req.model_name)
        except KeyError as e:
            raise HTTPException(
                status_code=404,
                detail=f"{e}. Available models: {registry.list_models()}",
            ) from e

        features = get_features_for_ids(feature_store, req.series_ids)
        found_ids = features["id"].to_list()
        missing_ids = set(req.series_ids) - set(found_ids)

        predictions_df = artifact.model.predict(horizon=req.horizon, ids=found_ids)

        # Group predictions by series_id
        series_responses = []
        for sid in found_ids:
            sid_preds = predictions_df.filter(pl.col("id") == sid)
            points = [
                PredictionPoint(date=row["date"], prediction=row["prediction"])
                for row in sid_preds.iter_rows(named=True)
            ]
            series_responses.append(
                PredictResponse(
                    series_id=sid,
                    model_name=artifact.model_name,
                    train_end=artifact.train_end,
                    predictions=points,
                )
            )

        # Surface missing ids in the response (won't fail the request)
        if missing_ids:
            for missing_id in missing_ids:
                series_responses.append(
                    PredictResponse(
                        series_id=missing_id,
                        model_name=artifact.model_name,
                        train_end=artifact.train_end,
                        predictions=[],  # empty = signals "not found"
                    )
                )

        return PredictBatchResponse(
            model_name=artifact.model_name,
            train_end=artifact.train_end,
            series=series_responses,
        )

    return app


# Convenience for `uvicorn forecasting.serving.api:app`
app = create_app()
