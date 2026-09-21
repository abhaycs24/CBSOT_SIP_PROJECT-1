import os
import sys
import pandas as pd
import numpy as np
import streamlit as st

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from telco_churn.serialization import ModelSerializer
from telco_churn.preprocessing import DataPreprocessor

st.set_page_config(
    page_title="Telco Churn & Segmentation Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telco Customer Segmentation & Churn Intelligence")
st.markdown("Predict subscriber churn risk in real-time and explore customer behavioral personas.")

# Sidebar Controls
st.sidebar.header("👤 Subscriber Profile Input")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure Months", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=20.0, max_value=120.0, value=65.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=20.0, max_value=10000.0, value=float(tenure * monthly_charges))
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Load model artifacts
serializer = ModelSerializer()
model, preprocessor = serializer.load()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Real-Time Churn Prediction Engine")
    if model is not None and preprocessor is not None:
        df_input = pd.DataFrame([{
            "Gender": gender,
            "Tenure Months": tenure,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
            "Contract": contract,
        }])
        
        df_clean = preprocessor.clean_data(df_input)
        _, X, _, X_scaled = preprocessor.encode_and_scale(df_clean)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)[:, 1][0]
        else:
            prob = 0.5

        st.metric(label="Churn Probability", value=f"{prob:.1%}")
        
        if prob >= 0.65:
            st.error("⚠️ High Risk Churn Customer - Immediate Retention Offer Advised!")
        elif prob >= 0.35:
            st.warning("⚡ Medium Risk Customer - Recommend Loyalty Discount.")
        else:
            st.success("✅ Low Risk Customer - High Subscriber Retention Stability.")
    else:
        st.info("Model artifacts not serialized yet. Please run `python src/telco_churn/cli.py` to train and save the model.")

with col2:
    st.subheader("📈 Pipeline Artifact Deliverables")
    artifact_choice = st.selectbox(
        "Select Artifact Visualization:",
        ["EDA Dashboard", "Customer Segmentation", "Model Performance", "Feature Importance"],
    )
    
    art_map = {
        "EDA Dashboard": "artifacts/eda_dashboard.png",
        "Customer Segmentation": "artifacts/customer_segmentation.png",
        "Model Performance": "artifacts/model_performance.png",
        "Feature Importance": "artifacts/feature_importance.png",
    }
    
    img_path = art_map[artifact_choice]
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)
    else:
        st.warning(f"Artifact plot {img_path} not generated yet.")
