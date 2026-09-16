"""Train and export the Telco churn model and K-Means customer segmentation.

This script extracts the modeling logic from ``CBSOT_SIP_PROJECT-1.ipynb`` so
the models can be served by ``app.py`` without running notebook cells:

* Cleaning: ``Total Charges`` blank strings -> NaN -> float -> median
  imputation; ``customerID`` dropped.
* Churn model: ``RandomForestClassifier(random_state=42)`` tuned with
  ``RandomizedSearchCV(n_iter=5, cv=3, random_state=42)`` on ordinal/label
  encoded features; target ``Churn Label`` (1 = churn), same 80/20 stratified
  split as the notebook (``random_state=42``).
* Segmentation: ``OrdinalEncoder`` -> ``StandardScaler`` -> ``KMeans(n_clusters=4,
  init='k-means++', random_state=42, n_init=10)`` fitted on the same feature set
  (all rows), as in the notebook (which label-encodes the frame before scaling).

The notebook's categorical ``LabelEncoder`` maps classes alphabetically; that
behaviour is reproduced with a per-column ``OrdinalEncoder`` inside the
exported pipeline (unknown categories at inference time map to ``-1`` instead
of raising, so the app degrades gracefully).

By default the notebook's synthetic placeholder dataset is used, seeded with
42 for reproducibility (the notebook itself does not seed it). Pass
``--data <csv>`` to train on a real export with the same column layout:
``customerID, Gender, Tenure Months, Monthly Charges, Total Charges, Contract,
Churn Label``.

Usage:
    python train.py [--data PATH] [--artifacts-dir PATH]
"""

from __future__ import annotations

import argparse
import datetime as _dt
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from model_utils import (
    ARTIFACT_DIR,
    CHURN_PIPELINE_FILE,
    DEFAULT_RISK_THRESHOLDS,
    METADATA_FILE,
    SEGMENT_PIPELINE_FILE,
)

TARGET_COLUMN = "Churn Label"
ID_COLUMNS = ["customerID"]
CATEGORICAL_FEATURES = ["Gender", "Contract"]
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
# Column order mirrors the notebook's X = df.drop(columns=['Churn Label']).
FEATURE_COLUMNS = [
    "Gender",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract",
]
N_CLUSTERS = 4
POSITIVE_CLASS = 1
RANDOM_STATE = 42


def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the training dataset (the notebook's synthetic sample by default)."""
    if data_path is not None:
        return pd.read_csv(data_path)
    return _build_notebook_sample()


def _build_notebook_sample() -> pd.DataFrame:
    """Seeded reproduction of the notebook's synthetic placeholder dataset."""
    rs = np.random.RandomState(RANDOM_STATE)
    data = {
        "customerID": [f"C{i}" for i in range(1, 101)],
        "Gender": rs.choice(["Male", "Female"], 100),
        "Tenure Months": rs.randint(1, 73, 100),
        "Monthly Charges": rs.uniform(20, 120, 100),
        "Total Charges": rs.uniform(50, 5000, 100).astype(str),
        "Contract": rs.choice(["Month-to-month", "One year", "Two year"], 100),
        "Churn Label": rs.choice([0, 1], 100),  # 0 for no churn, 1 for churn
    }
    df = pd.DataFrame(data)
    # Introduce the same ' ' blanks the notebook injects into 'Total Charges'.
    for _ in range(5):
        idx = rs.randint(0, len(df))
        df.loc[idx, "Total Charges"] = " "
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's cleaning steps and return a cleaned copy."""
    df = df.copy()
    if "Total Charges" in df.columns:
        df["Total Charges"] = df["Total Charges"].replace(" ", np.nan)
        # errors='coerce' also catches empty strings in real exports; for the
        # notebook data it behaves exactly like astype(float).
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())
    for column in ID_COLUMNS:
        if column in df.columns:
            df = df.drop(columns=[column])
    # The notebook runs LabelEncoder on the target; for the numeric 0/1 column
    # this is a no-op, but a 'Yes'/'No' export maps to No=0 / Yes=1 so that
    # class 1 always means "churned".
    if TARGET_COLUMN in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET_COLUMN]):
        df[TARGET_COLUMN] = LabelEncoder().fit_transform(df[TARGET_COLUMN])
    return df


def build_churn_preprocessor() -> ColumnTransformer:
    """Ordinal-encode categoricals (the notebook's LabelEncoder mapping) and
    pass numeric features through."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def train_churn_model(df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    """Tune the Random Forest exactly like the notebook and return a fitted
    ``preprocessor + model`` pipeline plus evaluation metrics.

    The pipeline is fitted on the notebook's 80% train split; the 20% test
    split stays untouched for the reported metrics.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_churn_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)

    param_dist = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        random_state=RANDOM_STATE,
    )
    search.fit(X_train_enc, y_train)

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("model", search.best_estimator_)]
    )
    # Refit end-to-end so the exported pipeline is a single, properly fitted
    # artifact (identical result: same split, params and random_state).
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, list(pipeline.classes_).index(POSITIVE_CLASS)]
    metrics = {
        "best_params": {
            key: (value if value is not None else "None")
            for key, value in search.best_params_.items()
        },
        "best_cv_score": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }
    return pipeline, metrics


def build_segment_preprocessor() -> ColumnTransformer:
    """Encoder used before scaling for K-Means.

    The notebook encodes the shared frame once (LabelEncoder) before both the
    clustering and classification branches, so segmentation reuses the exact
    same encoder as the churn preprocessor.
    """
    return build_churn_preprocessor()


def train_segment_model(df: pd.DataFrame) -> Pipeline:
    """Fit the ``encoder -> StandardScaler -> KMeans(k=4)`` pipeline on all
    rows, exactly as the notebook does before plotting the segmentation
    dashboard."""
    pipeline = Pipeline(
        [
            ("preprocessor", build_segment_preprocessor()),
            ("scaler", StandardScaler()),
            (
                "model",
                KMeans(
                    n_clusters=N_CLUSTERS,
                    init="k-means++",
                    random_state=RANDOM_STATE,
                    n_init=10,
                ),
            ),
        ]
    )
    pipeline.fit(df[FEATURE_COLUMNS])
    return pipeline


def compute_segment_profiles(
    df: pd.DataFrame, segment_pipeline: Pipeline
) -> dict[str, dict[str, Any]]:
    """Build deterministic, data-driven business profiles for each cluster.

    The notebook never interprets the clusters, so names are derived from each
    cluster's mean tenure vs. mean monthly charges relative to the overall
    medians (cluster labels themselves are arbitrary).
    """
    labels = np.asarray(segment_pipeline.named_steps["model"].labels_).astype(int)
    profiled = df.assign(_segment=labels)
    median_tenure = float(df["Tenure Months"].median())
    median_monthly = float(df["Monthly Charges"].median())

    profiles: dict[str, dict[str, Any]] = {}
    for segment in sorted(np.unique(labels)):
        rows = profiled[profiled["_segment"] == segment]
        mean_tenure = float(rows["Tenure Months"].mean())
        mean_monthly = float(rows["Monthly Charges"].mean())
        mean_total = float(rows["Total Charges"].mean())
        churn_rate = float(rows[TARGET_COLUMN].mean())
        top_contract = str(rows["Contract"].mode().iat[0])

        high_tenure = mean_tenure >= median_tenure
        high_spend = mean_monthly >= median_monthly
        if high_tenure and high_spend:
            name = "High-value loyal customers"
            recommendation = (
                "Reward loyalty with premium support, referral programs and "
                "bundled upgrades, and use these customers as advocates."
            )
        elif not high_tenure and high_spend:
            name = "New high-spend customers"
            recommendation = (
                "Protect early revenue with onboarding check-ins, an early "
                "service review and targeted retention offers."
            )
        elif high_tenure and not high_spend:
            name = "Long-tenure low-spend customers"
            recommendation = (
                "Educate these customers about higher-value plans and bundles "
                "and confirm satisfaction before pitching upgrades."
            )
        else:
            name = "New low-spend customers"
            recommendation = (
                "Run engagement campaigns, explain plan benefits, help with "
                "payment-method setup and monitor early churn signals."
            )

        description = (
            f"Avg tenure {mean_tenure:.0f} months, avg monthly charges "
            f"${mean_monthly:.2f}, avg total charges ${mean_total:.2f}. Most "
            f"common contract: {top_contract}. Training-data churn rate: "
            f"{churn_rate:.0%}."
        )
        profiles[str(int(segment))] = {
            "name": name,
            "description": description,
            "recommendation": recommendation,
            "avg_tenure_months": mean_tenure,
            "avg_monthly_charges": mean_monthly,
            "avg_total_charges": mean_total,
            "churn_rate": churn_rate,
            "customer_count": int(len(rows)),
        }
    return profiles


def build_metadata(
    df: pd.DataFrame,
    churn_pipeline: Pipeline,
    segment_pipeline: Pipeline,
    segments: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    data_description: str,
) -> dict[str, Any]:
    """Assemble the metadata artifact used for reliable inference and UI defaults."""
    return {
        "target_column": TARGET_COLUMN,
        "positive_class": POSITIVE_CLASS,
        "churn_features": list(FEATURE_COLUMNS),
        "segment_features": list(FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "n_clusters": N_CLUSTERS,
        "risk_thresholds": dict(DEFAULT_RISK_THRESHOLDS),
        "segments": dict(segments),
        "category_options": {
            column: sorted(df[column].dropna().unique().tolist())
            for column in CATEGORICAL_FEATURES
        },
        "input_defaults": {
            "Gender": str(df["Gender"].mode().iat[0]),
            "Contract": str(df["Contract"].mode().iat[0]),
            "Tenure Months": float(df["Tenure Months"].median()),
            "Monthly Charges": float(df["Monthly Charges"].median()),
            "Total Charges": float(df["Total Charges"].median()),
        },
        "input_ranges": {
            column: [float(df[column].min()), float(df[column].max())]
            for column in NUMERIC_FEATURES
        },
        "training_metrics": dict(metrics),
        "training_data": data_description,
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }


def validate_before_export(
    df: pd.DataFrame,
    churn_pipeline: Pipeline,
    segment_pipeline: Pipeline,
    n_clusters: int = N_CLUSTERS,
) -> None:
    """Fail fast (before saving) if the exported artifacts would be unusable."""
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Training data is missing required columns: {missing_columns}")

    if not hasattr(churn_pipeline, "predict_proba"):
        raise RuntimeError("Churn pipeline does not support predict_proba().")
    classes = list(np.asarray(churn_pipeline.classes_))
    if POSITIVE_CLASS not in classes:
        raise RuntimeError(
            f"Positive class {POSITIVE_CLASS} not found in trained classes {classes}."
        )

    kmeans = segment_pipeline.named_steps["model"]
    if kmeans.cluster_centers_.shape[0] != n_clusters:
        raise RuntimeError(
            f"KMeans produced {kmeans.cluster_centers_.shape[0]} clusters, expected {n_clusters}."
        )
    if len(np.unique(np.asarray(kmeans.labels_))) != n_clusters:
        raise RuntimeError(
            f"KMeans label count does not match the expected {n_clusters} clusters."
        )

    preprocessor = churn_pipeline.named_steps["preprocessor"]
    if list(getattr(preprocessor, "feature_names_in_", [])) != FEATURE_COLUMNS:
        raise RuntimeError("Fitted preprocessor feature columns differ from FEATURE_COLUMNS.")

    # A sample row must pass through the complete preprocessing + prediction
    # path of both pipelines.
    sample = df[FEATURE_COLUMNS].iloc[[0]]
    proba = churn_pipeline.predict_proba(sample)
    if proba.ndim != 2 or proba.shape[0] != 1 or proba.shape[1] != len(classes):
        raise RuntimeError(f"Unexpected predict_proba() output shape: {proba.shape}.")
    segment_id = int(segment_pipeline.predict(sample)[0])
    if not 0 <= segment_id < n_clusters:
        raise RuntimeError(f"Segment prediction {segment_id} is outside 0..{n_clusters - 1}.")


def save_artifacts(
    churn_pipeline: Pipeline,
    segment_pipeline: Pipeline,
    metadata: Mapping[str, Any],
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> list[Path]:
    """Persist all artifacts with joblib and verify they can be reloaded."""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        CHURN_PIPELINE_FILE: churn_pipeline,
        SEGMENT_PIPELINE_FILE: segment_pipeline,
        METADATA_FILE: metadata,
    }
    saved: list[Path] = []
    for filename, payload in payloads.items():
        path = directory / filename
        joblib.dump(payload, path)
        if not path.is_file():
            raise RuntimeError(f"Artifact was not written: {path}")
        joblib.load(path)  # reload check
        saved.append(path)
    return saved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and export the churn model and K-Means segmentation artifacts."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=(
            "Optional CSV with the notebook's column layout "
            "(customerID, Gender, Tenure Months, Monthly Charges, Total Charges, "
            "Contract, Churn Label). Defaults to the notebook's synthetic sample."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help=f"Output directory for the joblib artifacts (default: {ARTIFACT_DIR}).",
    )
    args = parser.parse_args(argv)

    try:
        raw_df = load_data(args.data)
        data_description = str(args.data) if args.data is not None else "synthetic notebook sample"
        df = clean_data(raw_df)
        print(f"Loaded data: {len(df)} rows ({data_description})")

        churn_pipeline, metrics = train_churn_model(df)
        print(
            "Churn model: RandomForestClassifier | best params: "
            f"{metrics['best_params']} | test ROC-AUC: {metrics['test_roc_auc']:.3f} "
            f"| test accuracy: {metrics['test_accuracy']:.3f}"
        )

        segment_pipeline = train_segment_model(df)
        print(f"Segmentation: KMeans (k={N_CLUSTERS}) on {FEATURE_COLUMNS}")

        profiles = compute_segment_profiles(df, segment_pipeline)
        metadata = build_metadata(
            df, churn_pipeline, segment_pipeline, profiles, metrics, data_description
        )

        validate_before_export(df, churn_pipeline, segment_pipeline)

        saved = save_artifacts(churn_pipeline, segment_pipeline, metadata, args.artifacts_dir)
        print("Segments:")
        for key, profile in profiles.items():
            print(f"  {key}: {profile['name']} ({profile['customer_count']} customers)")
        print(f"Artifacts written to {Path(args.artifacts_dir).resolve()}:")
        for path in saved:
            print(f"  - {path.name}")
        return 0
    except Exception as exc:  # keep the failure visible; exit non-zero
        print(f"Training/export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())




