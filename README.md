# Telco Customer Segmentation & Churn Prediction Pipeline

This project provides an end-to-end data science workflow on the Telco subscriber dataset. The objective is twofold: first, to group customers based on behavioral attributes using unsupervised learning, and second, to benchmark gradient boosting frameworks and predict potential churn using Bayesian hyperparameter optimization and SHAP explainability.

## Key Features & Highlights

1. **Modular Package Architecture**: Object-oriented Python package (`src/telco_churn/`) separating data ingestion, preprocessing, EDA, clustering, boosting models, SHAP explainability, and evaluation.
2. **Gradient Boosting Suite**: Comparative benchmarking across **Random Forest**, **XGBoost**, **LightGBM**, and **CatBoost**.
3. **Optuna Bayesian Hyperparameter Optimization**: Replaced standard grid searches with multi-trial Optuna study optimization targeting cross-validated ROC-AUC metrics.
4. **SHAP Model Interpretability**: Automated calculation of Shapley values producing global beeswarm summary plots and individual customer waterfall contribution charts.
5. **Config-Driven CLI Engine**: Run pipelines via command-line execution (`python -m telco_churn.cli`) with flexible YAML hyperparameter configurations (`configs/default.yaml`).

## Tech Stack Used
* **CLI Engine**: Python, PyYAML, Argparse
* **Data Core**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn, XGBoost, LightGBM, CatBoost
* **MLOps & Explainability**: Optuna, SHAP

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
│       ├── preprocessing.py # Preprocessing & feature scaling
│       ├── eda.py           # Exploratory data visualization
│       ├── clustering.py    # K-Means clustering & Elbow method
│       ├── modeling.py      # RF, XGBoost, LightGBM, CatBoost & Optuna tuning
│       ├── explainability.py# SHAP model interpretability
│       ├── evaluation.py    # Multi-model benchmark & plot generation
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
python -m src.telco_churn.cli --config configs/default.yaml
```
or set `PYTHONPATH=src`:
```bash
python -m telco_churn.cli --config configs/default.yaml
```

## Pipeline Deliverables & Artifacts
All generated visualizations are saved automatically in the `artifacts/` folder:
* `eda_dashboard.png`: Feature distributions and correlation heatmaps.
* `elbow_method.png`: Dynamic Elbow Method plotting for optimal cluster selection.
* `customer_segmentation.png`: Visualizing customer personas based on spend vs. loyalty.
* `model_benchmark.png`: Multi-model ROC-AUC overlay and performance comparison.
* `shap_summary.png`: Global SHAP feature impact beeswarm summary.
* `shap_waterfall.png`: Individual customer SHAP waterfall feature breakdown.
* `model_performance.png`: Best model Confusion Matrix and ROC-AUC Curve.
* `feature_importance.png`: Top 10 primary drivers of customer churn.
