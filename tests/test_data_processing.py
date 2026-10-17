import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd
from preprocessing import build_preprocessor


def test_preprocessor_transforms():
    df = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 3,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 0,
                "ca": 0,
                "thal": 1,
                "target": 1,
            }
        ]
    )
    X = df.drop(columns=["target"])
    pre = build_preprocessor()
    out = pre.fit_transform(X)
    assert out.shape[0] == 1
    assert out.shape[1] > 5


def test_data_schema():
    p = Path(__file__).resolve().parent.parent / "data" / "heart.csv"
    if not p.exists():
        import pytest

        pytest.skip("data not downloaded yet")
    df = pd.read_csv(p)
    assert set(["age", "chol", "target"]).issubset(df.columns)
    assert set(df["target"].unique()).issubset({0, 1})
