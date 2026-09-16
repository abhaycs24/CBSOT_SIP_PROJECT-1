"""Streamlit front end for the Telco churn model and K-Means segmentation.

Loads the joblib artifacts exported by ``train.py`` — no retraining and no
fitting happens here. Run from the repository root:

    streamlit run app.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from model_utils import (
    ARTIFACT_DIR,
    RISK_TIER_ADVICE,
    ArtifactError,
    build_input_dataframe,
    get_positive_class_probability,
    get_risk_tier,
    get_segment_details,
    load_artifacts,
)

st.set_page_config(
    page_title="Telco Churn & Segment Explorer",
    page_icon="📊",
    layout="centered",
)


@st.cache_resource
def load_artifacts_cached() -> dict[str, Any]:
    """Load the joblib artifacts once per process; they never change during a run."""
    return load_artifacts(ARTIFACT_DIR)


def _show_dev_details(exc: Exception) -> None:
    """Keep the UI readable for users while exposing the cause for developers."""
    with st.expander("Developer details"):
        st.code(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- artifacts
try:
    artifacts = load_artifacts_cached()
except FileNotFoundError as exc:
    st.error(
        "Model artifacts are missing. Generate them first by running:\n\n"
        "```bash\npython train.py\n```\n\n"
        "Then restart this app with `streamlit run app.py`."
    )
    _show_dev_details(exc)
    st.stop()
except ArtifactError as exc:
    st.error(
        "The model artifacts could not be loaded (corrupt or created with an "
        "incompatible library version). Regenerate them with `python train.py`."
    )
    _show_dev_details(exc)
    st.stop()

churn_pipeline = artifacts["churn_pipeline"]
segment_pipeline = artifacts["segment_pipeline"]
metadata = artifacts["metadata"]

category_options: dict[str, list] = metadata.get("category_options", {})
input_defaults: dict[str, Any] = metadata.get("input_defaults", {})
input_ranges: dict[str, list[float]] = metadata.get("input_ranges", {})

# ------------------------------------------------------------------- header
st.title("Telco Churn & Customer Segment Explorer")
st.caption(
    "This app loads the pre-trained artifacts from `artifacts/` — the models "
    "are **not** retrained here. Enter a customer profile in the sidebar and "
    "submit the form to see the model-estimated churn probability, risk tier "
    "and K-Means segment."
)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Customer information")

    gender_options = list(category_options.get("Gender", ["Female", "Male"]))
    contract_options = list(
        category_options.get("Contract", ["Month-to-month", "One year", "Two year"])
    )
    tenure_default = int(round(float(input_defaults.get("Tenure Months", 32))))
    monthly_default = float(input_defaults.get("Monthly Charges", 70.0))
    total_default = float(input_defaults.get("Total Charges", 2000.0))
    monthly_max = float(input_ranges.get("Monthly Charges", [0.0, 120.0])[1])
    total_max = float(input_ranges.get("Total Charges", [0.0, 5000.0])[1])

    with st.form("customer_inputs"):
        gender = st.selectbox(
            "Gender",
            gender_options,
            index=(
                gender_options.index(input_defaults["Gender"])
                if input_defaults.get("Gender") in gender_options
                else 0
            ),
        )
        tenure = st.slider(
            "Tenure (months)",
            min_value=0,
            max_value=max(72, tenure_default),
            value=min(tenure_default, max(72, tenure_default)),
            step=1,
        )
        monthly_charges = st.number_input(
            "Monthly charges ($)",
            min_value=0.0,
            max_value=max(200.0, monthly_max * 2.0),
            value=monthly_default,
            step=1.0,
        )
        total_charges = st.number_input(
            "Total charges ($)",
            min_value=0.0,
            max_value=max(10000.0, total_max * 2.0),
            value=total_default,
            step=10.0,
        )
        contract = st.selectbox(
            "Contract type",
            contract_options,
            index=(
                contract_options.index(input_defaults["Contract"])
                if input_defaults.get("Contract") in contract_options
                else 0
            ),
        )
        submitted = st.form_submit_button("Predict churn", type="primary")

# --------------------------------------------------------------- prediction
if not submitted:
    st.info("Fill in the customer details in the sidebar and press **Predict churn**.")
    st.stop()

values: dict[str, Any] = {
    "Gender": gender,
    "Tenure Months": int(tenure),
    "Monthly Charges": float(monthly_charges),
    "Total Charges": float(total_charges),
    "Contract": contract,
}

try:
    churn_frame = build_input_dataframe(values, metadata, feature_set="churn")
    segment_frame = build_input_dataframe(values, metadata, feature_set="segment")
except ValueError as exc:
    st.error(f"Input validation failed: {exc}")
    st.stop()

try:
    positive_class = int(metadata.get("positive_class", 1))
    probability = get_positive_class_probability(
        churn_pipeline, churn_frame, positive_class=positive_class
    )
    segment_id = int(segment_pipeline.predict(segment_frame)[0])
except AttributeError as exc:
    st.error(
        "The loaded model does not support `predict_proba()`, so a churn "
        "probability cannot be produced. Re-train with a probabilistic "
        "classifier: `python train.py`."
    )
    _show_dev_details(exc)
    st.stop()
except Exception as exc:
    st.error(
        "Prediction failed. The artifacts may be incompatible with the current "
        "library versions — regenerate them with `python train.py`."
    )
    _show_dev_details(exc)
    st.stop()

# ----------------------------------------------------------------- results
tier = get_risk_tier(probability, metadata.get("risk_thresholds"))
segment = get_segment_details(segment_id, metadata)

st.subheader("Churn prediction")
st.metric("Model-estimated churn probability", f"{probability * 100:.1f}%")
st.progress(
    float(min(max(probability, 0.0), 1.0)),
    text=f"Risk tier: **{tier}**",
)

if tier == "Low":
    st.success(f"**{tier} churn risk.** {RISK_TIER_ADVICE[tier]}")
elif tier == "Medium":
    st.warning(f"**{tier} churn risk.** {RISK_TIER_ADVICE[tier]}")
else:
    st.error(f"**{tier} churn risk.** {RISK_TIER_ADVICE[tier]}")

st.caption(
    "This probability is a model estimate based on historical patterns — "
    "it is not a guarantee of future behaviour."
)

st.subheader("Customer segment (K-Means)")
st.info(f"**{segment['name']}** — cluster {segment_id}\n\n{segment['description']}")
st.markdown(f"**Suggested retention actions**\n\n{segment['recommendation']}")
st.markdown(
    f"**Combined risk + segment suggestion ({tier} risk, {segment['name']}):**\n\n"
    f"{RISK_TIER_ADVICE[tier]} Additionally, for this segment: "
    f"{segment['recommendation']}"
)
st.caption(
    "Recommendations are business suggestions derived from the training-data "
    "segment profile and the predicted risk tier — not guaranteed outcomes."
)

with st.expander("Prediction details"):
    st.write("Churn probability:", float(probability))
    st.write("Risk tier:", tier)
    st.write("Segment id:", segment_id)
    st.write("Segment name:", segment["name"])
    st.dataframe(churn_frame)


