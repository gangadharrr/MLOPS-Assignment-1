"""Train, tune and compare classifiers; track everything in MLflow; save the best pipeline.

Method
------
* 80/20 stratified split. The 20% test set is touched once, for final reporting.
* Each candidate is tuned with GridSearchCV (inner 5-fold, scoring=ROC-AUC).
* Candidates are compared with *nested* CV (outer 5-fold around the grid search), so the
  number used to pick the model was not also used to tune its hyperparameters.
* The winner (highest nested-CV ROC-AUC) is refit on the full training split and saved.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import build_preprocessor  # noqa: E402

TARGET = "target"
SEED = 42
SCORING = ["accuracy", "precision", "recall", "roc_auc"]

# Hyperparameter grids (keys are Pipeline step__param names).
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=1),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [3, 5, 8, None],
            "clf__min_samples_leaf": [1, 3],
        },
    ),
}


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }


def save_plots(model, X_test, y_test, name, out):
    """Confusion matrix + ROC curve for the held-out test set; returns file paths."""
    cm_path, roc_path = out / f"confusion_{name}.png", out / f"roc_{name}.png"

    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt="d", ax=ax)
    ax.set_title(f"Confusion matrix - {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title(f"ROC curve - {name}")
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    return cm_path, roc_path


def main(data: str, out_dir: str, experiment: str = "heart-disease", figdir: str = "reports/figures"):
    df = pd.read_csv(data)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)

    mlflow.set_experiment(experiment)
    results, fitted = {}, {}

    for name, (clf, grid) in CANDIDATES.items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)

        # Nested CV: unbiased estimate of "tune + fit" performance on training data.
        nested = cross_validate(search, X_train, y_train, cv=outer, scoring=SCORING)
        cv_metrics = {f"cv_mean_{k}": float(nested[f"test_{k}"].mean()) for k in SCORING}
        cv_metrics.update({f"cv_std_{k}": float(nested[f"test_{k}"].std()) for k in SCORING})

        # Final tuned model on the whole training split, evaluated once on the test split.
        search.fit(X_train, y_train)
        best = search.best_estimator_
        test_metrics = {f"test_{k}": v for k, v in evaluate(best, X_test, y_test).items()}
        row = {**cv_metrics, **test_metrics}
        results[name], fitted[name] = row, (best, search.best_params_)
        print(f"\n== {name} ==\nbest params: {search.best_params_}")
        print({k: round(v, 4) for k, v in row.items() if not k.startswith("cv_std")})

        cm_path, roc_path = save_plots(best, X_test, y_test, name, out)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "seed": SEED, "grid": json.dumps(grid, default=str)})
            mlflow.log_params({k.replace("clf__", "best_"): v for k, v in search.best_params_.items()})
            mlflow.log_metrics(row)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")
            mlflow.log_artifact(str(roc_path), artifact_path="plots")
            mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(3))

    # Select on nested-CV ROC-AUC (training data only); test metrics are never used to choose.
    best_name = max(results, key=lambda n: results[n]["cv_mean_roc_auc"])
    best_model, best_params = fitted[best_name]
    joblib.dump(best_model, out / "model.pkl")
    summary = pd.DataFrame(results).T
    summary.to_csv(out / "metrics_summary.csv")
    (out / "best_params.json").write_text(
        json.dumps({"model": best_name, "params": best_params}, indent=2, default=str)
    )
    r = results[best_name]
    print(
        f"\nBest model: {best_name} (nested-CV ROC-AUC={r['cv_mean_roc_auc']:.4f}, "
        f"test ROC-AUC={r['test_roc_auc']:.4f}) -> {out / 'model.pkl'}"
    )

    with mlflow.start_run(run_name=f"best-{best_name}"):
        mlflow.log_param("best_model", best_name)
        mlflow.log_metrics({f"best_{k}": v for k, v in r.items()})
        mlflow.log_artifact(str(out / "metrics_summary.csv"))
        mlflow.log_artifact(str(out / "best_params.json"))
        mlflow.log_artifact(data, artifact_path="data")
        if Path(figdir).is_dir():  # EDA plots, if `src/eda.py` was run
            mlflow.log_artifacts(figdir, artifact_path="eda")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/heart.csv")
    ap.add_argument("--out", default="models")
    ap.add_argument("--experiment", default="heart-disease")
    ap.add_argument("--figdir", default="reports/figures")
    args = ap.parse_args()
    main(args.data, args.out, args.experiment, args.figdir)
