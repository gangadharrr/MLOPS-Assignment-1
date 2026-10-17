import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import MODEL_PATH, app  # noqa: E402

SAMPLE = json.loads((ROOT / "sample_request.json").read_text())


def _require_model():
    """Skip locally if untrained, but fail in CI so the predict path is always tested."""
    if not MODEL_PATH.exists():
        if os.environ.get("CI"):
            pytest.fail("models/model.pkl missing in CI; train before running tests")
        pytest.skip("model not trained yet")


@pytest.fixture()
def client():
    _require_model()
    with TestClient(app) as c:  # runs lifespan (model load)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_predict(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in ("disease", "no_disease")


@pytest.mark.parametrize(
    "field,value",
    [
        ("age", -5),
        ("sex", 7),
        ("cp", 0),
        ("slope", 0),
        ("thal", 1),
        ("ca", 9),
    ],
)
def test_predict_rejects_out_of_range(client, field, value):
    r = client.post("/predict", json={**SAMPLE, field: value})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "chol"}
    assert client.post("/predict", json=bad).status_code == 422


def test_metrics_counts_requests(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert 'api_requests_total{endpoint="/predict",status="200"}' in text
