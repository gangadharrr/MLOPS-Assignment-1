"""Download Heart Disease UCI (Cleveland) dataset with reproducibility."""

import argparse
import sys
from pathlib import Path
import pandas as pd

URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
]
COLUMNS = [
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
    "target",
]


def download(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, header=None, names=COLUMNS, na_values="?")
    return df


def main(out: str):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in URLS:
        try:
            print(f"Downloading from {url} ...")
            df = download(url)
            # Binarize target: 0 = no disease, 1-4 = disease -> 1
            df["target"] = (df["target"] > 0).astype(int)
            df.to_csv(out_path, index=False)
            print(f"Saved {df.shape} -> {out_path}")
            print(f"Class balance:\n{df['target'].value_counts(normalize=True)}")
            return
        except Exception as e:
            last_err = e
            print(f"Failed {url}: {e}", file=sys.stderr)
    raise SystemExit(f"All downloads failed: {last_err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/heart.csv")
    args = ap.parse_args()
    main(args.out)
