# Heart Disease MLOps — End-to-End (Assignment 1)

Predict heart-disease risk + serve as monitored API with full MLOps: EDA → training (MLflow) → tests → CI → Docker → K8s → Prometheus monitoring.

## Quickstart (clean setup)

```bash
pip install -r requirements.txt
python src/download_data.py --out data/heart.csv
python src/eda.py --data data/heart.csv --figdir reports/figures
python src/train.py --data data/heart.csv --out models
pytest -q
uvicorn app.main:app --port 8000
# in another terminal:
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
mlflow ui  # view experiment "heart-disease"
```

## Docker

`models/model.pkl` is git-ignored, so run `python src/train.py` before building (CI does this too).
The image installs only `requirements-serve.txt` and runs as a non-root user.

```bash
docker build -t heart-api:latest .
docker run -d -p 8000:8000 --name heart-api heart-api:latest
curl http://localhost:8000/ready
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_request.json
curl http://localhost:8000/metrics   # Prometheus
```

Expected `/predict` output for `sample_request.json` (first Cleveland row, true label 0):
`{"prediction":0,"probability":0.3572,"label":"no_disease"}`.
Verification transcripts: `screenshots/docker_verification.txt` (containerized build + all endpoints + 422 check)
and `screenshots/local_api_verification.txt` (non-Docker uvicorn run).

Input is validated (Cleveland coding: `cp` 1-4, `slope` 1-3, `ca` 0-3, `thal` in {3,6,7}, etc.); out-of-range values return HTTP 422.

## Kubernetes (Docker Desktop / Minikube / GKE / EKS / AKS)

```bash
docker build -t heart-api:latest .
# minikube: minikube image load heart-api:latest
kubectl apply -f k8s/
kubectl get pods,svc
curl http://<EXTERNAL-IP>/health
```

Manifests: `k8s/deployment.yaml` (2 replicas, probes, resources), `k8s/service.yaml` (LoadBalancer 80→8000).

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | readiness (model loaded) |
| POST | `/predict` | JSON patient → `{prediction, probability, label}` |
| GET | `/metrics` | Prometheus exposition |

Sample: `sample_request.json` (13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal).

## Monitoring (Prometheus + Grafana)

```bash
python src/train.py          # model.pkl must exist before the image build
docker compose up --build    # API :8000, Prometheus :9090, Grafana :3001 (admin/admin)
```

Prometheus and Grafana configs are baked into small images (`monitoring/prometheus.Dockerfile`,
`monitoring/grafana.Dockerfile`) instead of host bind mounts, so `docker compose up` works from any
checkout path. Grafana auto-loads the "Heart API" dashboard (request rate, error rate, p95 latency,
predictions by class); viewing needs no login, editing uses admin/admin. Generate traffic with a few
`curl` calls to `/predict`. Verified end to end — see `screenshots/monitoring_verification.txt`,
`screenshots/prometheus_targets.png`, `screenshots/grafana_dashboard.png`.

## Results (20% held-out test split)

Models are tuned with `GridSearchCV` (inner 5-fold, ROC-AUC) and compared with *nested* 5-fold CV on the
training split. The winner is the highest nested-CV ROC-AUC; the test split is never used to choose.

| Model | Nested-CV ROC-AUC (±sd) | Test acc | Prec | Recall | Test ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (selected, `models/model.pkl`) | **0.895** (±0.049) | 0.869 | 0.833 | 0.893 | 0.948 |
| Logistic Regression | 0.893 (±0.052) | 0.885 | 0.839 | 0.929 | 0.966 |

The two are statistically indistinguishable: the CV gap (0.0025) is far smaller than the fold-to-fold spread, and the
test set (61 rows) is too small to separate them either. Best parameters: `models/best_params.json`.

Artifacts: `models/metrics_summary.csv`, `models/confusion_*.png`, `models/roc_*.png`, MLflow experiment
`heart-disease` (params, metrics, plots, model with signature, EDA figures), `reports/figures/*.png`.

## Layout

```
src/download_data.py  src/eda.py  src/preprocessing.py  src/train.py  src/predict.py
app/main.py  tests/  Dockerfile  docker-compose.yml  .github/workflows/ci.yml
k8s/  monitoring/{prometheus.Dockerfile,grafana.Dockerfile,prometheus.yml,grafana/}  reports/  screenshots/  models/  data/
```

Offline inference: `python src/predict.py --json sample_request.json` or `--csv data/heart.csv --out preds.csv`.

## Architecture

```
CSV (UCI) → download_data → EDA (figures) → train (ColumnTransformer + grid-searched LR/RF, nested CV, MLflow)
  → model.pkl → FastAPI (/predict,/ready,/metrics, logging) → Docker → K8s LoadBalancer
  → Prometheus scrape (/metrics) → Grafana  |  CI: flake8 → train → pytest → docker build/smoke → artifacts
```

CI: `.github/workflows/ci.yml`. Monitoring: `prometheus_client` counters/histograms + request logs; Prometheus config in
`monitoring/prometheus.yml`, Grafana provisioning in `monitoring/grafana/`.
