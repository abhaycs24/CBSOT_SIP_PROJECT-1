# Telco Customer Segmentation & Churn Prediction Pipeline

This project provides an end-to-end data science workflow on the Telco subscriber dataset. The objective is twofold: first, to group customers based on behavioral attributes using unsupervised learning, and second, to engineer domain interaction features, balance minority churn classes using SMOTE-NC, validate data quality via Pydantic, and predict potential churn using a tuned machine learning model.

## Key Features & Highlights

1. **Pydantic Data Quality Validation**: Automated schema validation (`src/telco_churn/validation.py`) verifying incoming subscriber data types, numerical range bounds, and missing value constraints.
2. **Domain Feature Engineering Engine**: Custom transformer generating interaction features (`Tenure_Monthly_Ratio`, `Total_Charges_Dev`, `Contract_Risk_Index`).
3. **SMOTE-NC Class Imbalance Resampling**: Dynamic synthetic oversampling of minority churn cases across mixed continuous and nominal feature spaces (`src/telco_churn/sampling.py`).
4. **Modular Package Architecture**: Object-oriented Python package (`src/telco_churn/`) separating data ingestion, validation, feature engineering, preprocessing, EDA, clustering, modeling, and evaluation.
5. **Config-Driven CLI Engine**: Run pipelines via command-line execution (`python src/telco_churn/cli.py`) with flexible YAML hyperparameter configurations (`configs/default.yaml`).

## Tech Stack Used
* **CLI Engine**: Python, PyYAML, Argparse
* **Validation & Schema**: Pydantic v2
* **Feature Engineering & Imbalance**: Scikit-Learn, Imbalanced-Learn (SMOTE-NC)
* **Data Core**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn

## Directory Layout
```text
CBSOT_SIP_PROJECT-1/
├── configs/
│   └── default.yaml         # YAML Configuration settings
├── src/
│   └── telco_churn/         # Modular Package Source
│       ├── __init__.py
│       ├── config.py        # YAML configuration parser
│       ├── data.py          # Data ingestion and synthetic generator
│       ├── validation.py    # Pydantic schema validator
│       ├── feature_engineering.py # Domain feature transformer
│       ├── sampling.py      # SMOTE-NC class balancer
│       ├── preprocessing.py # Preprocessing & feature scaling
│       ├── eda.py           # Exploratory data visualization
│       ├── clustering.py    # K-Means clustering & Elbow method
│       ├── modeling.py      # Random Forest model & hyperparameter tuning
│       ├── evaluation.py    # Metrics report & plot generation
│       └── cli.py           # Command Line Interface runner
├── artifacts/               # Generated plot and report artifacts
├── CBSOT_SIP_PROJECT-1.ipynb # Original exploratory notebook
├── README.md
├── architecture.md
└── requirements.txt
```

## Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline via CLI
Execute the end-to-end workflow using the default YAML config:
```bash
python src/telco_churn/cli.py --config configs/default.yaml
```

## Pipeline Deliverables & Artifacts
All generated visualizations are saved automatically in the `artifacts/` folder:
* `eda_dashboard.png`: Feature distributions and correlation heatmaps.
* `elbow_method.png`: Dynamic Elbow Method plotting for optimal cluster selection.
* `customer_segmentation.png`: Visualizing customer personas based on spend vs. loyalty.
* `model_performance.png`: Confusion Matrix and ROC-AUC Curve analysis.
* `feature_importance.png`: Top 10 primary drivers of customer churn.
