"""Tests for the FastAPI prediction service.

Uses FastAPI's TestClient to exercise the endpoints without a running
server. We build the app pointing at a temp directory containing dummy
artifacts, so all tests are hermetic.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from forecasting.serving.feature_store import save_feature_store
from forecasting.serving.persistence import ModelArtifact, save_artifact


class FakeModel:
    """Stand-in for a real Forecaster, returns deterministic predictions."""

    def __init__(self, fixed_value: float = 5.0) -> None:
        self.fixed_value = fixed_value

    def predict(self, horizon: int, ids: list[str] | None = None) -> pl.DataFrame:
        ids = ids or ["A"]
        rows = []
        for sid in ids:
            for h in range(1, horizon + 1):
                rows.append(
                    {
                        "id": sid,
                        "date": date(2024, 1, 1) + timedelta(days=7 * h),
                        "prediction": self.fixed_value,
                    }
                )
        return pl.DataFrame(rows)


@pytest.fixture
def populated_artifacts_dir(tmp_path: Path) -> Path:
    """Create a temp artifacts dir with one fake model + feature store."""
    artifact = ModelArtifact(
        model=FakeModel(fixed_value=42.0),
        model_name="test-model",
        train_end=date(2024, 1, 1),
        feature_cols=["feature_1"],
        track_name="fresh_weekly",
    )
    save_artifact(artifact, tmp_path / "test-model.pkl")

    df = pl.DataFrame(
        {
            "id": ["A", "B", "C"],
            "date": [date(2024, 1, 1)] * 3,
            "sales": [10.0, 20.0, 30.0],
            "feature_1": [1.0, 2.0, 3.0],
        }
    )
    save_feature_store(
        df,
        tmp_path / "test-model_features.parquet",
        feature_cols=["feature_1"],
    )
    return tmp_path


@pytest.fixture
def client(populated_artifacts_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a TestClient against the populated artifacts dir."""
    monkeypatch.setenv("FORECASTING_ARTIFACTS_DIR", str(populated_artifacts_dir))
    # Re-import the module to pick up the new env var
    import importlib

    from forecasting.serving import api as api_module

    importlib.reload(api_module)
    return TestClient(api_module.app)


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["n_models_loaded"] == 1


class TestInfo:
    def test_info_lists_loaded_models(self, client: TestClient) -> None:
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 1
        assert body[0]["model_name"] == "test-model"
        assert body[0]["track_name"] == "fresh_weekly"
        assert body[0]["feature_count"] == 1


class TestPredict:
    def test_predict_known_series(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"series_id": "A", "horizon": 3, "model_name": "test-model"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["series_id"] == "A"
        assert len(body["predictions"]) == 3
        assert all(p["prediction"] == 42.0 for p in body["predictions"])

    def test_predict_unknown_series_404(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"series_id": "UNKNOWN", "horizon": 4, "model_name": "test-model"},
        )
        assert r.status_code == 404

    def test_predict_unknown_model_404(self, client: TestClient) -> None:
        r = client.post(
            "/predict",
            json={"series_id": "A", "horizon": 4, "model_name": "missing-model"},
        )
        assert r.status_code == 404

    def test_predict_invalid_horizon_422(self, client: TestClient) -> None:
        # horizon out of allowed range -> Pydantic validation error
        r = client.post(
            "/predict",
            json={"series_id": "A", "horizon": 100, "model_name": "test-model"},
        )
        assert r.status_code == 422


class TestPredictBatch:
    def test_batch_known_series(self, client: TestClient) -> None:
        r = client.post(
            "/predict-batch",
            json={"series_ids": ["A", "B"], "horizon": 2, "model_name": "test-model"},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["series"]) == 2
        sids = {s["series_id"] for s in body["series"]}
        assert sids == {"A", "B"}

    def test_batch_mixed_known_unknown(self, client: TestClient) -> None:
        """Unknown ids should appear in the response with empty predictions."""
        r = client.post(
            "/predict-batch",
            json={"series_ids": ["A", "UNKNOWN"], "horizon": 2, "model_name": "test-model"},
        )
        assert r.status_code == 200
        body = r.json()
        unknown_resp = next(s for s in body["series"] if s["series_id"] == "UNKNOWN")
        assert unknown_resp["predictions"] == []

    def test_batch_empty_list_422(self, client: TestClient) -> None:
        r = client.post(
            "/predict-batch",
            json={"series_ids": [], "horizon": 2, "model_name": "test-model"},
        )
        # Pydantic min_length=1 rejects this
        assert r.status_code == 422
