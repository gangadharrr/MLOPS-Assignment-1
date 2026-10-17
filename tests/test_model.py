import os
from pathlib import Path

import joblib
import pandas as pd
import pytest

MODEL = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


def test_model_artifact_loads_and_predicts():
    if not MODEL.exists():
        if os.environ.get("CI"):
            pytest.fail("models/model.pkl missing in CI; train before running tests")
        pytest.skip("model not trained yet")
    m = joblib.load(MODEL)
    row = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
            }
        ]
    )
    assert m.predict(row)[0] in (0, 1)
    proba = m.predict_proba(row)[0]
    assert 0.0 <= proba[1] <= 1.0
