# Final Report — Heart Disease MLOps (Assignment 1)

## 1. Setup
`pip install -r requirements.txt` (verified: Python 3.9 host, Docker `python:3.10-slim`, sklearn 1.6.1). Reproduce: download → EDA → train → pytest → `uvicorn app.main:app`. All scripts run from repo root with relative paths; `requirements.txt` pinned to tested set.

## 2. Data & EDA (303×14 Cleveland UCI, target binarized 0/1; 54.1% / 45.9%)
- Missing: `ca` 4, `thal` 2 → median/most-frequent imputation in pipeline.
- Figures (`reports/figures/`): `01_class_balance.png`, `02_histograms.png` (right-skew chol/oldpeak), `03_correlation.png` (thalach↔disease negative, exang/oldpeak positive), `04_age_chol.png`, `05_cp_target.png` (cp=typical angina strongly diseased).
- Download: `src/download_data.py` (UCI `processed.cleveland.data`, binarizes 1–4→1).

## 3. Features & Models
`ColumnTransformer`: 5 numeric (median+StandardScaler) + 8 categorical (most-frequent+OneHot). Preprocessing lives inside each `Pipeline`, so training and serving apply the same transforms:
- Logistic Regression — linear baseline; grid: `C` in {0.01, 0.1, 1, 10, 100}.
- Random Forest — nonlinear interactions; grid: `n_estimators` {100, 300} x `max_depth` {3, 5, 8, None} x `min_samples_leaf` {1, 3}.

**Tuning and selection.** Stratified 80/20 split. Each model is tuned with `GridSearchCV` (inner 5-fold, ROC-AUC). Models are compared with nested 5-fold CV (outer folds wrap the grid search), so the score used to choose a model was not also used to tune it. The winner is the highest nested-CV ROC-AUC; the test split is reported once and never used for selection.

| Model | Nested-CV ROC-AUC (±sd) | Nested-CV acc | Test acc | Test prec | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest (selected) | 0.895 (±0.049) | 0.797 | 0.869 | 0.833 | 0.893 | 0.948 |
| Logistic Regression | 0.893 (±0.052) | 0.830 | 0.885 | 0.839 | 0.929 | 0.966 |

Best params: RF `max_depth=5, min_samples_leaf=3, n_estimators=100`; LR `C=1`. **Caveat:** the models are effectively tied. The nested-CV gap (0.0025) is much smaller than the fold-to-fold sd (~0.05), and 61 test rows cannot separate them either (LR scores higher on every test metric, but that was not used to choose). The selection rule picked RF; for a screening use-case where recall matters, LR is an equally defensible choice.

Artifacts: `models/model.pkl` (full pipeline), `metrics_summary.csv`, `best_params.json`, `confusion_*.png`, `roc_*.png`.

## 4. Experiment Tracking
MLflow experiment `heart-disease`, one run per model plus a `best-<model>` summary run. Each model run logs: best hyperparameters and the search grid, nested-CV metrics (mean and sd) and test metrics, confusion-matrix and ROC plots, and the sklearn pipeline with an input example (signature inferred). The summary run logs the metrics CSV, `best_params.json`, the dataset, and the EDA figures. Compare runs with `mlflow ui`.

## 5. Packaging
`joblib` pipeline (preprocessing+model in one object), pinned `requirements.txt`, versioned `Dockerfile`. Inference needs only `model.pkl` + JSON.

## 6. CI/CD & Tests
`tests/` (20 passing locally with `CI=1`): schema + transform, imputation and unseen-category handling, metric computation, an end-to-end `train.main` run on synthetic data (artifacts + selection rule), inference helper, API health/ready/predict, 422 on out-of-range and missing fields, metrics counting. With `CI` set, a missing model fails the tests instead of skipping them.
`.github/workflows/ci.yml` (not yet run on GitHub; no repo pushed yet): flake8 → train (on committed `data/heart.csv`) → pytest → docker build → container smoke test (`/ready`, `/predict`) → upload `models/`. Any failing step fails the workflow.

## 7. Container
FastAPI: `/health` (liveness), `/ready` (model loaded; also the k8s readiness probe), `/predict` (validated 13 fields → `{prediction, probability, label}`), `/metrics`. The model loads at startup, so a container without `model.pkl` fails to start. Request logs record method, path, status and latency only, not patient data. Metrics: `api_requests_total{endpoint,status}` (includes 422/5xx), `api_latency_seconds`, `api_predictions_total{label}`.
Verified locally with uvicorn (`screenshots/local_api_verification.txt`): `/predict` on the sample returns `{"prediction":0,"probability":0.3572,"label":"no_disease"}`. The slimmed Docker image (serve-only deps, non-root) was NOT re-verified after the last change because Docker Desktop stopped responding; an earlier, heavier image did build and serve. Rebuild and capture screenshots before submitting.

## 8. Deployment
`k8s/deployment.yaml` (replicas 2, readiness `/ready`, liveness `/health`, requests/limits) + `service.yaml` (LoadBalancer 80→8000). YAML-parse validated; cluster apply is `kubectl apply -f k8s/` (note: no live cluster in this sandbox — `kubectl` has no server — use Docker Desktop/Minikube/GKE per README). Monitoring: `docker-compose.yml` runs the API with Prometheus (scrapes `heart-api:8000/metrics`) and Grafana with a provisioned "Heart API" dashboard (request rate, error rate, p95 latency, predictions by class). The compose file, Prometheus config and dashboard JSON are syntax-validated but have not been run end to end, because Docker was unavailable.

## 9. Architecture
`UCI → download → EDA → train+MLflow → model.pkl → FastAPI → Docker → K8s LB → Prometheus; CI guards every push.`

## 10. Reproduce & Submit
Repo contains code, Dockerfile, requirements, data+script, figures, tests, workflow, manifests, screenshots, this report + `ASSIGNMENT_INSTRUCTIONS.md`. Video: run Quickstart → `docker run` curls → `kubectl apply` → `mlflow ui` screen-record. Public URL: deploy image to GKE/ACR + expose LB, paste external IP here.
