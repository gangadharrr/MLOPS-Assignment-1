# MLOps Assignment 1 — Instructions (Markdown Conversion)

> Converted from `MLOps-Assignment-1-Instructions.pdf` for easy reference. Original PDF retained as source of truth.

**Course:** Machine Learning Operations (MLOps) — AIMLCZG523
**Assignment:** 01 | **Total Marks:** 50
**Title:** MLOps Experimental Learning Assignment: End-to-End ML Model Development, CI/CD, and Production Deployment

## Objective

Design, develop, and deploy a scalable, reproducible ML solution using modern MLOps best practices. Emphasis on practical automation, experiment tracking, CI/CD pipelines, containerization, cloud deployment, and monitoring — mirroring real-world production scenarios.

## Dataset: Heart Disease UCI

- **Source:** UCI Machine Learning Repository
- **CSV** containing 14+ features (`age`, `sex`, blood pressure, cholesterol, etc.) and a binary target (presence/absence of heart disease).

## Problem Statement

Build an ML classifier to predict risk of heart disease based on patient health data, and deploy the solution as a cloud-ready, monitored API.

## Tasks & Marks

| # | Task | Marks |
|---|------|-------|
| 1 | Data Acquisition & EDA — download script, cleaning, preprocessing, professional visualizations (histograms, correlation heatmaps, class balance) | 5 |
| 2 | Feature Engineering & Model Development — scaling/encoding, ≥2 classifiers (e.g., Logistic Regression + Random Forest), tuning documented, CV + metrics (accuracy, precision, recall, ROC-AUC) | 8 |
| 3 | Experiment Tracking — MLflow (or similar): log params, metrics, artifacts, plots | 5 |
| 4 | Model Packaging & Reproducibility — reusable format (MLflow/pickle/ONNX), `requirements.txt`/`env.yml`, preprocessing pipeline for full reproducibility | 7 |
| 5 | CI/CD & Automated Testing — unit tests (pytest), GitHub Actions/Jenkins: lint + test + train, artifacts/logs per run | 8 |
| 6 | Containerization — Docker for model-serving API (Flask/FastAPI), `/predict` JSON in → prediction + confidence out, must build & run locally | 5 |
| 7 | Production Deployment — public cloud or local K8s (GKE/EKS/AKS/Minikube/Docker Desktop), deployment manifest or Helm, LoadBalancer/Ingress, verify + screenshots | 7 |
| 8 | Monitoring & Logging — API request logging, simple monitoring (Prometheus+Grafana or metrics/logs dashboard) | 3 |
| 9 | Documentation & Reporting — Markdown/PDF report: setup, EDA + modelling choices, experiment summary, architecture diagram, CI/CD + deployment screenshots, repo link | 2 |

## Deliverables

- GitHub repo with: code, `Dockerfile(s)`, `requirements.txt`/`env.yml`, cleaned dataset + download script, notebooks/scripts (EDA, training, inference), `tests/` with unit tests, Actions workflow YAML / Jenkinsfile, deployment manifests/Helm, screenshot folder.
- Final written report (~10 pages, doc/docx/pdf).
- Deployed API URL (public) or local access instructions.
- Short video of overall pipeline.

## Production-Readiness Requirements

- All scripts execute from clean setup via requirements file.
- Model serves correctly in isolated Docker container (build/test proof required).
- Pipeline fails on code/test errors with clear logs.

## Academic Integrity

Independent work required. No direct copying of code, notebooks, visualizations, feature engineering, CI/CD workflows, deployment configs, or pipelines. Highly similar submissions subject to review/penalties. Personalize approach, experimentation, architecture, monitoring, and docs.
