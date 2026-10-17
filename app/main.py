"""FastAPI heart-disease serving API with request logging + Prometheus metrics.

Feature coding follows the UCI Cleveland file:
  sex 0/1, cp 1-4, fbs 0/1, restecg 0-2, exang 0/1, slope 1-3, ca 0-3, thal in {3, 6, 7}.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("heart-api")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"
_model = None


def get_model():
    """Load the model once; raise if the artifact is missing."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")
        _model = joblib.load(MODEL_PATH)
        log.info("Loaded model from %s", MODEL_PATH)
    return _model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Fail fast: a container without a model should not start.
    get_model()
    yield


class Patient(BaseModel):
    age: float = Field(..., ge=1, le=120, examples=[63])
    sex: int = Field(..., ge=0, le=1, examples=[1])
    cp: int = Field(..., ge=1, le=4, examples=[1])
    trestbps: float = Field(..., ge=50, le=300, examples=[145])
    chol: float = Field(..., ge=50, le=700, examples=[233])
    fbs: int = Field(..., ge=0, le=1, examples=[1])
    restecg: int = Field(..., ge=0, le=2, examples=[2])
    thalach: float = Field(..., ge=40, le=250, examples=[150])
    exang: int = Field(..., ge=0, le=1, examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, examples=[2.3])
    slope: int = Field(..., ge=1, le=3, examples=[3])
    ca: int = Field(..., ge=0, le=3, examples=[0])
    thal: int = Field(..., examples=[6])

    @field_validator("thal")
    @classmethod
    def thal_valid(cls, v: int) -> int:
        if v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v


FEATURES = list(Patient.model_fields)

REQUESTS = Counter("api_requests_total", "Total API requests", ["endpoint", "status"])
LATENCY = Histogram("api_latency_seconds", "Request latency", ["endpoint"])
PREDICTIONS = Counter("api_predictions_total", "Predictions by class", ["label"])

app = FastAPI(title="Heart Disease API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Count and time every request (including errors and 422s); log without payloads."""
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        latency = time.perf_counter() - t0
        path = request.url.path
        if path != "/metrics":
            REQUESTS.labels(endpoint=path, status=str(status)).inc()
            LATENCY.labels(endpoint=path).observe(latency)
        log.info("%s %s -> %s %.3fs", request.method, path, status, latency)


@app.get("/health")
def health():
    """Liveness: process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: model artifact is loadable."""
    try:
        get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "ready"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(p: Patient):
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    row = pd.DataFrame([{k: getattr(p, k) for k in FEATURES}])
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1])
    label = "disease" if pred == 1 else "no_disease"
    PREDICTIONS.labels(label=label).inc()
    log.info("prediction=%s probability=%.4f", label, proba)
    return {"prediction": pred, "probability": round(proba, 4), "label": label}
