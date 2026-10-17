"""Batch / single-record inference using the saved pipeline (no server needed).

Usage:
    python src/predict.py --json sample_request.json
    python src/predict.py --csv data/heart.csv --out predictions.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def load_model(path: str = "models/model.pkl"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run `python src/train.py` first.")
    return joblib.load(p)


def predict_frame(model, df: pd.DataFrame) -> pd.DataFrame:
    """Return prediction + probability for each row; raises if feature columns are missing."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[FEATURES]
    out = pd.DataFrame(
        {
            "prediction": model.predict(X).astype(int),
            "probability": model.predict_proba(X)[:, 1].round(4),
        },
        index=df.index,
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="JSON file with a single patient record")
    src.add_argument("--csv", help="CSV file with one patient per row")
    ap.add_argument("--model", default="models/model.pkl")
    ap.add_argument("--out", help="write predictions to this CSV (CSV mode)")
    args = ap.parse_args()

    model = load_model(args.model)
    if args.json:
        df = pd.DataFrame([json.loads(Path(args.json).read_text())])
    else:
        df = pd.read_csv(args.csv)
    res = predict_frame(model, df)
    if args.out:
        res.to_csv(args.out, index=False)
        print(f"Wrote {len(res)} predictions -> {args.out}")
    else:
        print(res.to_string())


if __name__ == "__main__":
    main()
