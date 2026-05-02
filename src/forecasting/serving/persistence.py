"""Save and load trained Forecaster instances.

Why pickle:
For this project's scale, pickle is the simplest format that works for
both LightGBM (which has its own native format but plays well with pickle)
and statsforecast/AutoETS (which depends on numpy arrays inside Python
objects, hard to express in any non-pickle format). Pickle has known
security caveats (loading untrusted pickle = code execution), so we
restrict it to artifacts produced by our own training pipeline.

Why a tiny wrapper class:
A bare pickle file is opaque. We wrap it with metadata (model name,
training date, train_end date, feature columns) so the API can verify
compatibility before serving.

Versioning:
The `format_version` field protects future-us if we change the artifact
schema. Loading code refuses unknown versions explicitly rather than
silently loading garbage.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# Bump this when the artifact schema changes incompatibly
ARTIFACT_FORMAT_VERSION = "1.0"


@dataclass
class ModelArtifact:
    """A saved model + its metadata.

    Attributes:
        model: The trained Forecaster instance (LightGBMForecaster,
            HoltWintersBaseline, etc.).
        model_name: Friendly name for logging and API responses.
        train_end: Last date in the training data. Used by the API to
            decide whether re-training is needed.
        feature_cols: Feature column names the model expects. Allows the
            API to validate inputs.
        track_name: Which track the model was trained on (fresh_weekly, etc.).
        trained_at: When the artifact was created.
        format_version: Schema version. Loading code checks this.
        extra: Free-form dict for model-specific metadata (hyperparameters,
            training metrics, etc.)
    """

    model: Any
    model_name: str
    train_end: date
    feature_cols: list[str]
    track_name: str
    trained_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    format_version: str = ARTIFACT_FORMAT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


def save_artifact(artifact: ModelArtifact, path: str | Path) -> Path:
    """Save a ModelArtifact to disk as a pickle file.

    Args:
        artifact: The artifact to save.
        path: File path (any extension; .pkl by convention).

    Returns:
        The Path the artifact was written to.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
    return p


def load_artifact(path: str | Path) -> ModelArtifact:
    """Load a ModelArtifact from disk, validating its format version.

    Args:
        path: Path to the artifact file.

    Returns:
        The loaded ModelArtifact.

    Raises:
        FileNotFoundError: if the file doesn't exist.
        ValueError: if the artifact's format_version is incompatible.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No artifact at {p}")

    with p.open("rb") as f:
        artifact = pickle.load(f)

    if not isinstance(artifact, ModelArtifact):
        raise ValueError(
            f"File at {p} does not contain a ModelArtifact (got {type(artifact).__name__})"
        )

    if artifact.format_version != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"Artifact format version {artifact.format_version} is not "
            f"compatible with current loader (expected {ARTIFACT_FORMAT_VERSION}). "
            f"Re-train the model with the current pipeline."
        )

    return artifact
