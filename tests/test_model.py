import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.loader import load_raw_data
from src.data.preprocessor import clean_data, get_features_and_target
from src.features.engineering import engineer_features
from src.models.evaluator import business_cost


@pytest.fixture
def model():
    path = Path("models/xgb_v1.pkl")
    if not path.exists():
        pytest.skip("Model not trained yet")
    return joblib.load(path)


@pytest.fixture
def X_y():
    df = clean_data(load_raw_data())
    df = engineer_features(df)
    return get_features_and_target(df)


def test_model_loads(model):
    assert model is not None


def test_model_predicts_proba(model, X_y):
    X, y = X_y
    probs = model.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_model_predicts_binary(model, X_y):
    X, y = X_y
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_model_recall_above_threshold(model, X_y):
    from sklearn.metrics import recall_score
    X, y = X_y
    preds = model.predict(X)
    recall = recall_score(y, preds)
    assert recall >= 0.65, f"Recall {recall:.3f} is below 0.65 threshold"


def test_model_auc_above_threshold(model, X_y):
    from sklearn.metrics import roc_auc_score
    X, y = X_y
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, probs)
    assert auc >= 0.80, f"AUC {auc:.3f} is below 0.80 threshold"


def test_business_cost_structure(model, X_y):
    X, y = X_y
    preds = model.predict(X)
    result = business_cost(y, preds)
    assert "total_cost_ils" in result
    assert "saved_revenue_ils" in result
    assert result["saved_revenue_ils"] > 0
    assert result["true_positives"] + result["false_negatives"] == y.sum()