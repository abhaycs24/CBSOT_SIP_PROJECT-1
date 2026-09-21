# Telco Customer Segmentation & Churn Prediction Pipeline

This project provides an end-to-end data science workflow on the Telco subscriber dataset. The objective is twofold: first, to benchmark multi-algorithm customer segmentation (K-Means, GMM, HDBSCAN) using quantitative Silhouette metrics and UMAP 2D projections, and second, to predict potential churn using a tuned machine learning model.

## Key Features & Highlights

1. **Multi-Algorithm Customer Segmentation**: Comparative benchmarking across **K-Means**, **Gaussian Mixture Models (GMM)**, and **HDBSCAN**.
2. **Quantitative Clustering Validation**: Automated calculation of **Silhouette Coefficient**, **Calinski-Harabasz Index**, and **Davies-Bouldin Index**.
3. **UMAP 2D Dimensionality Reduction**: Non-linear manifold projections (`umap_clusters.png`) visualizing customer persona separation in high-dimensional feature space.
4. **Modular Package Architecture**: Object-oriented Python package (`src/telco_churn/`) separating data ingestion, preprocessing, EDA, multi-algorithm clustering, modeling, and evaluation.
5. **Config-Driven CLI Engine**: Run pipelines via command-line execution (`python src/telco_churn/cli.py`) with flexible YAML hyperparameter configurations (`configs/default.yaml`).

## Tech Stack Used
* **CLI Engine**: Python, PyYAML, Argparse
* **Clustering & Projections**: Scikit-Learn, HDBSCAN, UMAP-learn
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
│       ├── preprocessing.py # Preprocessing & feature scaling
│       ├── eda.py           # Exploratory data visualization
│       ├── clustering.py    # Multi-algorithm clustering, Silhouette & UMAP
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
* `umap_clusters.png`: UMAP 2D non-linear cluster projection visual dashboard.
* `model_performance.png`: Confusion Matrix and ROC-AUC Curve analysis.
* `feature_importance.png`: Top 10 primary drivers of customer churn.
