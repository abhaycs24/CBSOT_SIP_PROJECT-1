"""Unit and artifact tests for the churn/segmentation export pipeline.

The artifact tests execute ``train.main()`` against a temporary directory, so
they exercise the full train -> save -> load -> predict path without touching
the repository's ``artifacts/`` folder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import train as train_module
from model_utils import (
    CHURN_PIPELINE_FILE,
    METADATA_FILE,
    SEGMENT_PIPELINE_FILE,
    ArtifactError,
    build_input_dataframe,
    get_positive_class_probability,
    get_risk_tier,
    get_segment_details,
    load_artifacts,
)


class _FakeProbabilisticModel:
    """Minimal stand-in exposing classes_ + predict_proba()."""

    def __init__(self, classes, proba_row):
        self.classes_ = np.asarray(classes)
        self._proba_row = np.asarray(proba_row, dtype=float)

    def predict_proba(self, model_input):
        return self._proba_row.reshape(1, -1)


class _NoProbaModel:
    """Stand-in for a model without predict_proba()."""

    classes_ = np.asarray([0, 1])

    def predict(self, model_input):
        return np.asarray([0])


# ------------------------------------------------------- risk-tier thresholds
@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.0, "Low"),
        (0.32, "Low"),
        (0.33, "Medium"),  # boundary: Low is strictly below low_max
        (0.5, "Medium"),
        (0.659, "Medium"),
        (0.66, "High"),  # boundary: Medium is strictly below medium_max
        (1.0, "High"),
    ],
)
def test_get_risk_tier_default_thresholds(probability, expected):
    assert get_risk_tier(probability) == expected


def test_get_risk_tier_custom_thresholds():
    thresholds = {"low_max": 0.6, "medium_max": 0.8}
    assert get_risk_tier(0.5, thresholds) == "Low"
    assert get_risk_tier(0.7, thresholds) == "Medium"
    assert get_risk_tier(0.9, thresholds) == "High"


def test_get_risk_tier_invalid_thresholds():
    with pytest.raises(ValueError):
        get_risk_tier(0.5, {"low_max": 0.8, "medium_max": 0.4})


# ------------------------------------------------- positive-class probability
def test_positive_class_probability_uses_classes_order():
    # Class 1 at index 1.
    model = _FakeProbabilisticModel(classes=[0, 1], proba_row=[0.25, 0.75])
    assert get_positive_class_probability(model, None) == pytest.approx(0.75)

    # Class 1 at index 0 — index-based extraction would return the wrong value.
    model = _FakeProbabilisticModel(classes=[1, 0], proba_row=[0.9, 0.1])
    assert get_positive_class_probability(model, None) == pytest.approx(0.9)


def test_missing_predict_proba_raises_clear_error():
    with pytest.raises(AttributeError, match="predict_proba"):
        get_positive_class_probability(_NoProbaModel(), None)


def test_positive_class_not_in_trained_classes():
    model = _FakeProbabilisticModel(classes=[0, 2], proba_row=[0.5, 0.5])
    with pytest.raises(ValueError, match="Positive class"):
        get_positive_class_probability(model, None)


# -------------------------------------------------------- input-data building
def _metadata():
    return {
        "churn_features": ["Gender", "Tenure Months", "Contract"],
        "segment_features": ["Gender", "Tenure Months", "Contract"],
        "numeric_features": ["Tenure Months"],
    }


def test_build_input_dataframe_column_order_and_target_excluded():
    values = {
        "Contract": "Two year",
        "Gender": "Female",
        "Tenure Months": 12,
        "Churn Label": 1,  # must be ignored
    }
    frame = build_input_dataframe(values, _metadata())
    assert list(frame.columns) == ["Gender", "Tenure Months", "Contract"]
    assert len(frame) == 1
    assert "Churn Label" not in frame.columns
    assert frame["Tenure Months"].dtype.kind in "if"  # numeric coercion applied


def test_build_input_dataframe_missing_feature_raises():
    values = {"Gender": "Female"}  # Tenure Months and Contract missing
    with pytest.raises(ValueError, match="Missing required input"):
        build_input_dataframe(values, _metadata())


def test_build_input_dataframe_invalid_numeric_raises():
    values = {"Gender": "Female", "Tenure Months": "not-a-number", "Contract": "One year"}
    with pytest.raises(ValueError):
        build_input_dataframe(values, _metadata())


# ------------------------------------------------------------- segment lookup
def test_get_segment_details_known_mapping():
    metadata = {
        "segments": {
            "2": {
                "name": "High-value loyal customers",
                "description": "High tenure, high spend.",
                "recommendation": "Offer loyalty rewards.",
            }
        }
    }
    details = get_segment_details(2, metadata)
    assert details["name"] == "High-value loyal customers"
    assert details["recommendation"] == "Offer loyalty rewards."


def test_get_segment_details_unknown_cluster_neutral_label():
    details = get_segment_details(7, {"segments": {}})
    assert details["name"] == "Segment 7"
    assert details["recommendation"]  # a non-empty generic recommendation exists


# ------------------------------------------------ artifact training & loading
@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Train once into a module-scoped temp dir and return (dir, artifacts)."""
    artifact_dir = tmp_path_factory.mktemp("artifacts_module")
    exit_code = train_module.main(["--artifacts-dir", str(artifact_dir)])
    assert exit_code == 0
    return artifact_dir, load_artifacts(artifact_dir)


def test_train_creates_all_artifacts(trained_artifacts):
    artifact_dir, artifacts = trained_artifacts
    for filename in (CHURN_PIPELINE_FILE, SEGMENT_PIPELINE_FILE, METADATA_FILE):
        assert (artifact_dir / filename).is_file(), f"missing artifact: {filename}"
    assert set(artifacts) == {"churn_pipeline", "segment_pipeline", "metadata"}


def test_metadata_contains_required_keys(trained_artifacts):
    _, artifacts = trained_artifacts
    metadata = artifacts["metadata"]
    assert metadata["target_column"] == train_module.TARGET_COLUMN
    assert metadata["churn_features"] == train_module.FEATURE_COLUMNS
    assert metadata["segment_features"] == train_module.FEATURE_COLUMNS
    assert metadata["n_clusters"] == train_module.N_CLUSTERS == 4
    assert metadata["risk_thresholds"] == {"low_max": 0.33, "medium_max": 0.66}
    assert set(metadata["segments"]) == {"0", "1", "2", "3"}
    for profile in metadata["segments"].values():
        assert profile["name"] and profile["recommendation"]


def test_churn_pipeline_supports_predict_proba(trained_artifacts):
    _, artifacts = trained_artifacts
    assert hasattr(artifacts["churn_pipeline"], "predict_proba")
    assert 1 in list(np.asarray(artifacts["churn_pipeline"].classes_))


def test_required_input_columns_match_feature_lists(trained_artifacts):
    _, artifacts = trained_artifacts
    metadata = artifacts["metadata"]
    assert set(metadata["churn_features"]) == set(train_module.FEATURE_COLUMNS)
    assert set(metadata["numeric_features"]) == set(train_module.NUMERIC_FEATURES)
    assert set(metadata["categorical_features"]) == set(train_module.CATEGORICAL_FEATURES)


# ------------------------------------------------------ end-to-end prediction
def test_end_to_end_sample_prediction(trained_artifacts):
    """One valid customer row must reach both prediction paths."""
    _, artifacts = trained_artifacts
    metadata = artifacts["metadata"]
    defaults = metadata["input_defaults"]

    values = {
        "Gender": defaults["Gender"],
        "Tenure Months": defaults["Tenure Months"],
        "Monthly Charges": defaults["Monthly Charges"],
        "Total Charges": defaults["Total Charges"],
        "Contract": defaults["Contract"],
    }
    churn_frame = build_input_dataframe(values, metadata, feature_set="churn")
    segment_frame = build_input_dataframe(values, metadata, feature_set="segment")

    probability = get_positive_class_probability(
        artifacts["churn_pipeline"], churn_frame, positive_class=metadata["positive_class"]
    )
    assert 0.0 <= probability <= 1.0

    tier = get_risk_tier(probability, metadata["risk_thresholds"])
    assert tier in {"Low", "Medium", "High"}

    segment_id = int(artifacts["segment_pipeline"].predict(segment_frame)[0])
    assert 0 <= segment_id < metadata["n_clusters"]

    details = get_segment_details(segment_id, metadata)
    assert details["name"]
    assert details["recommendation"]


# ------------------------------------------------------ missing/corrupt files
def test_load_artifacts_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="python train.py"):
        load_artifacts(tmp_path)


def test_load_artifacts_corrupt(tmp_path):
    (tmp_path / CHURN_PIPELINE_FILE).write_bytes(b"not-a-joblib-file")
    (tmp_path / SEGMENT_PIPELINE_FILE).write_bytes(b"not-a-joblib-file")
    (tmp_path / METADATA_FILE).write_bytes(b"not-a-joblib-file")
    with pytest.raises(ArtifactError, match="could not be loaded"):
        load_artifacts(tmp_path)


def test_load_artifacts_incomplete_metadata(tmp_path, trained_artifacts):
    import joblib

    source_dir, artifacts = trained_artifacts
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    joblib.dump(artifacts["churn_pipeline"], broken_dir / CHURN_PIPELINE_FILE)
    joblib.dump(artifacts["segment_pipeline"], broken_dir / SEGMENT_PIPELINE_FILE)
    joblib.dump({"unrelated": True}, broken_dir / METADATA_FILE)

    with pytest.raises(ArtifactError, match="missing required key"):
        load_artifacts(broken_dir)


def test_clean_data_imputes_blank_total_charges():
    raw = train_module._build_notebook_sample()
    cleaned = train_module.clean_data(raw)
    assert "customerID" not in cleaned.columns
    assert cleaned["Total Charges"].isna().sum() == 0
    assert cleaned["Total Charges"].dtype.kind == "f"
    assert set(cleaned["Churn Label"].unique()) <= {0, 1}

