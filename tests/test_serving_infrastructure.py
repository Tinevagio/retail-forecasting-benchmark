"""Tests for the serving infrastructure (persistence + feature store)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from forecasting.serving.feature_store import (
    get_features_for_ids,
    load_feature_store,
    save_feature_store,
)
from forecasting.serving.persistence import (
    ARTIFACT_FORMAT_VERSION,
    ModelArtifact,
    load_artifact,
    save_artifact,
)

# ---- A simple object to use as a "model" in tests --------------------------


class FakeModel:
    """Minimal pickleable stand-in for a real Forecaster."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "id": ["X"] * horizon,
                "date": [date(2024, 1, 1) + timedelta(days=7 * h) for h in range(horizon)],
                "prediction": [self.value] * horizon,
            }
        )


# ---- Persistence tests -----------------------------------------------------


class TestSaveLoadArtifact:
    def test_round_trip(self, tmp_path: Path) -> None:
        original = ModelArtifact(
            model=FakeModel(value=42.0),
            model_name="test-model",
            train_end=date(2020, 1, 1),
            feature_cols=["lag_1", "month"],
            track_name="fresh_weekly",
        )
        path = tmp_path / "artifact.pkl"
        save_artifact(original, path)

        loaded = load_artifact(path)
        assert loaded.model_name == "test-model"
        assert loaded.train_end == date(2020, 1, 1)
        assert loaded.feature_cols == ["lag_1", "month"]
        assert loaded.track_name == "fresh_weekly"
        assert loaded.format_version == ARTIFACT_FORMAT_VERSION
        # The model object survives pickling
        assert loaded.model.value == 42.0

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "new_dir" / "nested" / "artifact.pkl"
        artifact = ModelArtifact(
            model=FakeModel(1.0),
            model_name="m",
            train_end=date(2020, 1, 1),
            feature_cols=[],
            track_name="t",
        )
        save_artifact(artifact, path)
        assert path.exists()

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_artifact(tmp_path / "nonexistent.pkl")

    def test_load_wrong_object_raises(self, tmp_path: Path) -> None:
        import pickle

        path = tmp_path / "wrong.pkl"
        with path.open("wb") as f:
            pickle.dump({"not": "an artifact"}, f)

        with pytest.raises(ValueError, match="does not contain a ModelArtifact"):
            load_artifact(path)

    def test_load_incompatible_version_raises(self, tmp_path: Path) -> None:
        import pickle

        path = tmp_path / "old.pkl"
        old = ModelArtifact(
            model=FakeModel(1.0),
            model_name="m",
            train_end=date(2020, 1, 1),
            feature_cols=[],
            track_name="t",
        )
        # Manually downgrade the version to simulate an old artifact
        old.format_version = "0.1"
        with path.open("wb") as f:
            pickle.dump(old, f)

        with pytest.raises(ValueError, match="format version"):
            load_artifact(path)


# ---- Feature store tests --------------------------------------------------


def _make_df_with_features() -> pl.DataFrame:
    """3 SKUs, 5 dates each, 2 features."""
    rows = []
    week_dates = [date(2020, 1, 6) + timedelta(days=7 * i) for i in range(5)]
    for sku in ["A", "B", "C"]:
        for d in week_dates:
            rows.append(
                {
                    "id": sku,
                    "date": d,
                    "sales": 10.0,
                    "feature_1": 1.5,
                    "feature_2": 3.0,
                }
            )
    return pl.DataFrame(rows)


class TestSaveLoadFeatureStore:
    def test_only_latest_per_id_saved(self, tmp_path: Path) -> None:
        df = _make_df_with_features()
        path = tmp_path / "fs.parquet"
        save_feature_store(df, path)

        loaded = load_feature_store(path)
        # 3 SKUs, one row each
        assert loaded.height == 3
        assert set(loaded["id"].unique().to_list()) == {"A", "B", "C"}
        # All should have the latest date
        latest_date = df["date"].max()
        assert (loaded["date"] == latest_date).all()

    def test_filter_columns(self, tmp_path: Path) -> None:
        df = _make_df_with_features()
        path = tmp_path / "fs.parquet"
        save_feature_store(df, path, feature_cols=["feature_1"])

        loaded = load_feature_store(path)
        # Only id, date, feature_1 — feature_2 and sales should be excluded
        assert "feature_2" not in loaded.columns
        assert "sales" not in loaded.columns
        assert "feature_1" in loaded.columns

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        df = _make_df_with_features()
        with pytest.raises(ValueError, match="Missing columns"):
            save_feature_store(df, tmp_path / "fs.parquet", feature_cols=["nonexistent"])

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_feature_store(tmp_path / "nope.parquet")


class TestGetFeaturesForIds:
    def test_filters_correctly(self, tmp_path: Path) -> None:
        df = _make_df_with_features()
        path = tmp_path / "fs.parquet"
        save_feature_store(df, path)
        fs = load_feature_store(path)

        result = get_features_for_ids(fs, ["A", "C"])
        assert set(result["id"].to_list()) == {"A", "C"}
        assert result.height == 2

    def test_missing_ids_silently_dropped(self, tmp_path: Path) -> None:
        df = _make_df_with_features()
        path = tmp_path / "fs.parquet"
        save_feature_store(df, path)
        fs = load_feature_store(path)

        # Request 2 known + 1 unknown id
        result = get_features_for_ids(fs, ["A", "Z", "B"])
        # Only A and B should be returned
        assert set(result["id"].to_list()) == {"A", "B"}
