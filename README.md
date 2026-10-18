# Customer Churn Prediction and Explainable AI

## Overview

This project explores customer churn prediction using machine learning. It trains a Random Forest classifier and uses SHAP (SHapley Additive exPlanations) to explain the model's predictions.

The project includes a Streamlit dashboard for viewing global feature importance and understanding individual customer predictions.

**Note:** The model in this prototype was trained on synthetic data. Its results are illustrative and should not be interpreted as evidence about real customers.

## Features

- Customer churn prediction using a Random Forest classifier.
- Global feature importance using mean absolute SHAP values.
- Individual customer explanations showing features that push predictions toward or away from churn.
- Interactive Streamlit dashboard.
- Automated tests for model predictions, saved artifacts, and SHAP values.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SHAP
- Streamlit
- Matplotlib
- Joblib
- Pytest

## Project Structure

```text
CBSOT_SIP_PROJECT-1/
├── app.py
├── CBSOT_SIP_PROJECT-1.ipynb
├── churn_model.pkl
├── feature_names.pkl
├── X_test.pkl
├── churn_shap_values.pkl
├── requirements.txt
├── README.md
└── tests/
    └── test_churn_model.py
```

## Installation

1. Clone the repository and open its project directory.
2. Create and activate a Python virtual environment if desired.
3. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Dashboard

Run the following command from the project root:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

### Dashboard Pages

**Overview:** Displays model information and the number of test customers.

**Global Feature Importance:** Ranks features by their mean absolute SHAP values.

**Individual Customer:** Displays a selected customer's predicted churn probability, prediction, feature values, and SHAP contributions.

## Running Tests

Run the automated tests from the project root:

```bash
python -m pytest tests/test_churn_model.py -v -p no:cacheprovider
```

The tests check that required artifacts exist, the saved model can make predictions, feature names match the test data, and SHAP values have valid dimensions and values.

## Understanding SHAP

SHAP values help explain how individual features influence a model's prediction relative to its baseline. Positive values in this dashboard push the prediction toward churn, while negative values push it away from churn.

SHAP explains the model's behavior; it does not establish that a feature causes customer churn.

## Limitations

- The current model uses synthetic data.
- Some categorical features are represented by encoded numeric values.
- The explanations reflect the trained model and dataset and may not generalize to real customer populations.

## Future Improvements

- Display human-readable categorical feature values.
- Add more comprehensive model and explanation tests.
- Improve model artifact generation and reproducibility.
- Evaluate the approach on a suitable real-world dataset, subject to its licensing and privacy requirements.
