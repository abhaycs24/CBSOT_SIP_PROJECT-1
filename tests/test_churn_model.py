
import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_model_files_exist():
    """Check that the trained model and its supporting files exist."""
    assert (PROJECT_ROOT / "churn_model.pkl").exists()
    assert (PROJECT_ROOT / "feature_names.pkl").exists()
    assert (PROJECT_ROOT / "X_test.pkl").exists()
    assert (PROJECT_ROOT / "churn_shap_values.pkl").exists()


def test_model_can_make_predictions():
    """Check that the saved model predicts churn for test customers."""
    model = joblib.load(PROJECT_ROOT / "churn_model.pkl")
    X_test = joblib.load(PROJECT_ROOT / "X_test.pkl")

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_feature_names_match_test_data():
    """Check that the saved feature names match the test dataset."""
    feature_names = joblib.load(PROJECT_ROOT / "feature_names.pkl")
    X_test = joblib.load(PROJECT_ROOT / "X_test.pkl")

    assert isinstance(feature_names, list)
    assert list(X_test.columns) == feature_names



import numpy as np


def test_shap_values_match_test_data():
    """Check that SHAP explanations cover all test customers and features."""
    shap_values = joblib.load(PROJECT_ROOT / "churn_shap_values.pkl")
    X_test = joblib.load(PROJECT_ROOT / "X_test.pkl")

    if isinstance(shap_values, list):
        churn_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        churn_values = shap_values[:, :, 1]
    else:
        churn_values = shap_values

    assert churn_values.shape[0] == len(X_test)
    assert churn_values.shape[1] == X_test.shape[1]
    assert np.isfinite(churn_values).all()


def test_global_shap_importance_is_valid():
    """Check that global SHAP importance contains valid values."""
    shap_values = joblib.load(PROJECT_ROOT / "churn_shap_values.pkl")

    if isinstance(shap_values, list):
        churn_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        churn_values = shap_values[:, :, 1]
    else:
        churn_values = shap_values

    importance = np.abs(churn_values).mean(axis=0)

    assert len(importance) == len(
        joblib.load(PROJECT_ROOT / "feature_names.pkl")
    )
    assert np.isfinite(importance).all()
    assert (importance >= 0).all()
