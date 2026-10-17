"""Unit tests for training/evaluation/inference code, on a small synthetic dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import predict  # noqa: E402
import train  # noqa: E402
from preprocessing import build_preprocessor  # noqa: E402


def make_df(n=80, seed=0):
    """Synthetic data with the real schema/coding; target depends on a few features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(30, 77, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.integers(94, 200, n),
            "chol": rng.integers(126, 400, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 202, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0, 6, n).round(1),
            "slope": rng.integers(1, 4, n),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3.0, 6.0, 7.0], n),
        }
    )
    score = (df.cp >= 4).astype(int) + (df.exang == 1).astype(int) + (df.oldpeak > 2).astype(int)
    df["target"] = (score + rng.integers(0, 2, n) >= 2).astype(int)
    return df


def test_preprocessor_imputes_missing_values():
    df = make_df().drop(columns="target")
    df.loc[0, "ca"] = np.nan
    df.loc[1, "thal"] = np.nan
    df.loc[2, "chol"] = np.nan
    Z = build_preprocessor().fit_transform(df)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    assert np.isfinite(Z).all()
    assert Z.shape[0] == len(df)


def test_preprocessor_ignores_unseen_category_at_inference():
    df = make_df().drop(columns="target")
    pre = build_preprocessor().fit(df)
    odd = df.head(1).copy()
    odd["thal"] = 99.0
    Z = pre.transform(odd)
    Z = Z.toarray() if hasattr(Z, "toarray") else Z
    assert np.isfinite(Z).all()


def test_evaluate_returns_valid_metrics():
    df = make_df()
    X, y = df.drop(columns="target"), df.target
    from sklearn.pipeline import Pipeline

    model = Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    m = train.evaluate(model, X, y)
    assert set(m) == {"accuracy", "precision", "recall", "roc_auc"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_train_main_writes_artifacts_and_picks_by_cv(tmp_path, monkeypatch):
    data = tmp_path / "heart.csv"
    make_df(120).to_csv(data, index=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    # Shrink the grids so the test is fast.
    small = {
        "logistic_regression": (LogisticRegression(max_iter=500), {"clf__C": [0.1, 1]}),
        "random_forest": (
            train.CANDIDATES["random_forest"][0],
            {"clf__n_estimators": [20], "clf__max_depth": [3]},
        ),
    }
    monkeypatch.setattr(train, "CANDIDATES", small)

    out = tmp_path / "models"
    train.main(str(data), str(out), experiment="unit-test", figdir=str(tmp_path / "nofigs"))

    for f in ["model.pkl", "metrics_summary.csv", "best_params.json"]:
        assert (out / f).exists(), f
    summary = pd.read_csv(out / "metrics_summary.csv", index_col=0)
    best = summary["cv_mean_roc_auc"].idxmax()
    import json

    assert json.loads((out / "best_params.json").read_text())["model"] == best


def test_predict_frame_and_missing_columns():
    df = make_df()
    X, y = df.drop(columns="target"), df.target
    from sklearn.pipeline import Pipeline

    model = Pipeline([("pre", build_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]).fit(X, y)
    res = predict.predict_frame(model, X.head(5))
    assert list(res.columns) == ["prediction", "probability"]
    assert len(res) == 5 and res.prediction.isin([0, 1]).all()
    with pytest.raises(ValueError, match="Missing feature columns"):
        predict.predict_frame(model, X.drop(columns="chol"))


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_model(str(tmp_path / "nope.pkl"))
