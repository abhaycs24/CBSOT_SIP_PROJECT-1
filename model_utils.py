"""Reusable inference helpers shared by ``train.py``, ``app.py`` and the tests.

This module is intentionally free of Streamlit imports so every helper can be
unit-tested in isolation (see ``tests/test_model_utils.py``). It is the single
source of truth for artifact paths/names and the default risk thresholds;
``train.py`` exports them and ``app.py`` consumes them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

# Paths are anchored to this file so the app works regardless of the current
# working directory.
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

CHURN_PIPELINE_FILE = "churn_pipeline.joblib"
SEGMENT_PIPELINE_FILE = "segment_pipeline.joblib"
METADATA_FILE = "metadata.joblib"

#: Default churn-risk tier boundaries, applied when the metadata artifact does
#: not define any. Tiers: Low < low_max <= Medium < medium_max <= High.
DEFAULT_RISK_THRESHOLDS: dict[str, float] = {"low_max": 0.33, "medium_max": 0.66}

#: Deterministic risk-tier retention guidance. It is combined with the
#: data-driven per-segment recommendation stored in the metadata artifact.
RISK_TIER_ADVICE: dict[str, str] = {
    "Low": (
        "Offer loyalty benefits, referral programs and cross-sell "
        "opportunities, and keep monitoring satisfaction."
    ),
    "Medium": (
        "Run engagement campaigns, educate the customer about their plan, "
        "assist with payment methods and monitor the account closely."
    ),
    "High": (
        "Reach out proactively, review the current services, offer targeted "
        "discounts or contract-renewal incentives and arrange a technical "
        "support follow-up."
    ),
}


class ArtifactError(RuntimeError):
    """Raised when artifacts exist but cannot be deserialized or are incompatible."""


def get_risk_tier(probability: float, thresholds: Mapping[str, float] | None = None) -> str:
    """Map a churn probability to a ``Low``/``Medium``/``High`` risk tier.

    ``thresholds`` may override ``low_max`` and/or ``medium_max``; anything not
    provided falls back to :data:`DEFAULT_RISK_THRESHOLDS`.
    """
    bounds = {**DEFAULT_RISK_THRESHOLDS, **dict(thresholds or {})}
    low_max = float(bounds["low_max"])
    medium_max = float(bounds["medium_max"])
    if low_max > medium_max:
        raise ValueError("risk threshold 'low_max' must not exceed 'medium_max'")
    if probability < low_max:
        return "Low"
    if probability < medium_max:
        return "Medium"
    return "High"


def get_positive_class_probability(
    model: Any,
    model_input: Any,
    positive_class: int = 1,
) -> float:
    """Return the ``predict_proba()`` value for the positive (churn) class.

    The class column is located via ``model.classes_`` rather than assuming
    index 1, so the result stays correct even if the estimator was fitted with
    a different class ordering.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The loaded model does not implement predict_proba(); a churn "
            "probability cannot be produced. Re-train with a probabilistic "
            "classifier first: python train.py"
        )
    proba = np.asarray(model.predict_proba(model_input))[0]
    classes = np.asarray(getattr(model, "classes_", np.arange(len(proba))))
    matches = np.flatnonzero(classes == positive_class)
    if matches.size == 0:
        raise ValueError(
            f"Positive class {positive_class!r} is not among the trained "
            f"classes {classes.tolist()}."
        )
    return float(proba[matches[0]])


def build_input_dataframe(
    values: Mapping[str, Any],
    metadata: Mapping[str, Any],
    feature_set: str = "churn",
) -> pd.DataFrame:
    """Build a one-row DataFrame with exactly the training feature columns.

    Only the columns listed in ``metadata[f"{feature_set}_features"]`` are read
    from ``values`` — everything else (including the target column) is ignored,
    so the target can never leak into inference. Column order is controlled by
    the metadata, and numeric columns are coerced to numeric dtype to match
    what the fitted preprocessor expects.
    """
    feature_key = f"{feature_set}_features"
    features = list(metadata.get(feature_key, []))
    if not features:
        raise ValueError(f"Metadata is missing '{feature_key}'.")
    missing = [column for column in features if column not in values]
    if missing:
        raise ValueError(f"Missing required input(s): {', '.join(missing)}")
    frame = pd.DataFrame([{column: values[column] for column in features}], columns=features)
    for column in metadata.get("numeric_features", []):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame


def get_segment_details(segment_id: Any, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the human-readable profile for a K-Means cluster id.

    Cluster labels are arbitrary; the business meaning comes from the
    ``metadata["segments"]`` mapping exported by ``train.py``. Unknown cluster
    ids fall back to a neutral label instead of failing.
    """
    try:
        key = str(int(segment_id))
    except (TypeError, ValueError):
        key = str(segment_id)
    stored = metadata.get("segments", {}).get(key)
    if stored is not None:
        details = dict(stored)
    else:
        details = {
            "name": f"Segment {key}",
            "description": (
                "This cluster has no stored business profile; segment "
                "interpretation requires labelling it against the training "
                "cluster statistics."
            ),
            "recommendation": (
                "Review this cluster's profile in the training data before "
                "targeting it with specific retention actions."
            ),
        }
    details.setdefault("name", f"Segment {key}")
    details.setdefault("description", "")
    details.setdefault("recommendation", "")
    return details


def load_artifacts(artifact_dir: str | Path = ARTIFACT_DIR) -> dict[str, Any]:
    """Load the churn pipeline, segment pipeline and metadata artifact.

    Raises ``FileNotFoundError`` (with the recovery command) when files are
    missing, and :class:`ArtifactError` when they exist but cannot be loaded.
    """
    directory = Path(artifact_dir)
    paths = {
        "churn_pipeline": directory / CHURN_PIPELINE_FILE,
        "segment_pipeline": directory / SEGMENT_PIPELINE_FILE,
        "metadata": directory / METADATA_FILE,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Model artifacts are missing: " + ", ".join(missing)
            + ". Generate them first with:  python train.py"
        )
    try:
        artifacts = {name: joblib.load(path) for name, path in paths.items()}
    except Exception as exc:  # surfaced to the caller with full context attached
        raise ArtifactError(
            "Model artifacts could not be loaded (corrupt file or created with "
            f"an incompatible library version): {exc}"
        ) from exc

    metadata = artifacts["metadata"]
    if not isinstance(metadata, Mapping):
        raise ArtifactError("metadata.joblib does not contain a metadata mapping.")
    for key in ("churn_features", "segment_features", "target_column"):
        if key not in metadata:
            raise ArtifactError(
                f"Metadata artifact is missing required key '{key}'. "
                "Re-run:  python train.py"
            )
    return artifacts

