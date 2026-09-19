import pandas as pd
import matplotlib.pyplot as plt


def get_risk_level(probability):
    """Convert churn probability into a risk category."""
    if probability >= 0.75:
        return "High"
    elif probability >= 0.50:
        return "Medium"
    return "Low"


def create_churn_risk_table(model, X_test, customer_ids):
    """
    Generate customer-level churn probabilities and risk levels.

    Parameters:
        model: Trained churn prediction model.
        X_test: Test features used by the model.
        customer_ids: Customer IDs corresponding to X_test.

    Returns:
        DataFrame sorted by churn probability in descending order.
    """

    churn_probability = model.predict_proba(X_test)[:, 1]

    risk_levels = [
        get_risk_level(probability)
        for probability in churn_probability
    ]

    risk_df = pd.DataFrame({
        "Customer ID": customer_ids.loc[X_test.index].values,
        "Churn Probability": churn_probability,
        "Risk Level": risk_levels
    })

    return risk_df.sort_values(
        by="Churn Probability",
        ascending=False
    )


def get_high_risk_customers(risk_df):
    """Return customers classified as high churn risk."""
    return risk_df[risk_df["Risk Level"] == "High"]


def plot_risk_distribution(risk_df):
    """Plot the distribution of customer churn risk levels."""

    risk_counts = (
        risk_df["Risk Level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    risk_counts.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title("Customer Churn Risk Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Customers")

    plt.tight_layout()
    plt.show()