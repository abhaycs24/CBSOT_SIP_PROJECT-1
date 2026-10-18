
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Customer Churn Explainability",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Explainability")
st.write("Understand which features influence customer churn predictions.")


@st.cache_resource
def load_files():
    model = joblib.load("churn_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    X_test = joblib.load("X_test.pkl")
    shap_values = joblib.load("churn_shap_values.pkl")

    return model, feature_names, X_test, shap_values


try:
    model, feature_names, X_test, shap_values = load_files()
except FileNotFoundError:
    st.error(
        "Required model or SHAP files are missing. "
        "Run the notebook first to generate them."
    )
    st.stop()


# Select SHAP values for the churn class.
if isinstance(shap_values, list):
    churn_values = shap_values[1]

elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    churn_values = shap_values[:, :, 1]

else:
    churn_values = shap_values


st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a view",
    [
        "Overview",
        "Global Feature Importance",
        "Individual Customer"
    ]
)


# --------------------------------------------------
# PAGE 1: OVERVIEW
# --------------------------------------------------
if page == "Overview":
    st.subheader("Model Overview")

    st.success(
        "Random Forest model and SHAP data loaded successfully."
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "Number of features",
        len(feature_names)
    )

    col2.metric(
        "Test customers available",
        len(X_test)
    )

    st.markdown("### About this dashboard")

    st.write(
        """
        This dashboard uses SHAP (SHapley Additive exPlanations)
        to explain predictions from a Random Forest model.
        """
    )

    st.markdown(
        """
        **Dashboard features**
        - Global Feature Importance: see which features influence
          model predictions most across the test dataset.
        - Individual Customer: inspect a customer's prediction
          and the factors pushing it toward or away from churn.
        """
    )

    st.warning(
        "The model was trained on synthetic data. Results are "
        "illustrative and should not be treated as evidence about "
        "real customers."
    )


# --------------------------------------------------
# PAGE 2: GLOBAL FEATURE IMPORTANCE
# --------------------------------------------------
elif page == "Global Feature Importance":
    st.subheader("Global Feature Importance")

    importance = np.abs(churn_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Mean Absolute SHAP Value": importance
    }).sort_values(
        "Mean Absolute SHAP Value",
        ascending=True
    )

    st.write(
        "Higher mean absolute SHAP values indicate greater "
        "influence on the model's predictions."
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        importance_df["Feature"],
        importance_df["Mean Absolute SHAP Value"]
    )

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_ylabel("Feature")
    ax.set_title("Global Feature Importance")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Feature Importance Ranking")

    st.dataframe(
        importance_df.sort_values(
            "Mean Absolute SHAP Value",
            ascending=False
        ),
        width="stretch"
    )


# --------------------------------------------------
# PAGE 3: INDIVIDUAL CUSTOMER EXPLANATION
# --------------------------------------------------
elif page == "Individual Customer":
    st.subheader("Individual Customer Explanation")

    customer_index = st.selectbox(
        "Select a test customer",
        options=list(range(len(X_test))),
        format_func=lambda i: f"Customer {i + 1}"
    )

    customer = X_test.iloc[[customer_index]]

    prediction = int(model.predict(customer)[0])
    probabilities = model.predict_proba(customer)[0]

    churn_probability = probabilities[1]

    col1, col2 = st.columns(2)

    col1.metric(
        "Predicted churn probability",
        f"{churn_probability:.1%}"
    )

    col2.metric(
        "Model prediction",
        "Churn" if prediction == 1 else "No churn"
    )

    if prediction == 1:
        st.warning("The model predicts that this customer may churn.")
    else:
        st.success("The model predicts that this customer may stay.")

    st.markdown("### Customer Feature Values")

    st.dataframe(
        customer,
        width="stretch"
    )

    customer_shap = churn_values[customer_index]

    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "Feature Value": customer.iloc[0].to_numpy(),
        "SHAP Value": customer_shap
    })

    explanation_df["Effect"] = np.where(
        explanation_df["SHAP Value"] > 0,
        "Pushes toward churn",
        np.where(
            explanation_df["SHAP Value"] < 0,
            "Pushes away from churn",
            "No contribution"
        )
    )

    explanation_df = explanation_df.sort_values(
        "SHAP Value",
        key=abs,
        ascending=False
    )

    st.markdown("### Factors Influencing This Prediction")

    st.dataframe(
        explanation_df,
        width="stretch"
    )

    # Visualize each feature's SHAP contribution.
    plot_df = explanation_df.sort_values(
        "SHAP Value",
        ascending=True
    )

    colors = [
        "tomato" if value > 0 else "steelblue"
        for value in plot_df["SHAP Value"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        plot_df["Feature"],
        plot_df["SHAP Value"],
        color=colors
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8
    )

    ax.set_xlabel(
        "SHAP Value (positive pushes toward churn)"
    )

    ax.set_ylabel("Feature")
    ax.set_title("Individual Customer SHAP Explanation")

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Red bars push the model prediction toward churn; blue bars "
        "push it away from churn. Feature values may be numerically "
        "encoded. SHAP explains model behavior, not causation."
    )
