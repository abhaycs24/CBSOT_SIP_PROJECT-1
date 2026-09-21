# System Architecture Diagram

```text
               [ configs/default.yaml ]
                          │
                          ▼
            ┌───────────────────────────┐
            │   CLI Engine (cli.py)     │
            └─────────────┬─────────────┘
                          │
                          ▼
             [ Data Module (data.py) ]
           (Synthetic Data / CSV Ingestion)
                          │
                          ▼
         [ Preprocessing (preprocessing.py) ]
    (String Sanitization, Median Imputation, Scaling)
                          │
                          ▼
  ┌───────────────────────┴───────────────────────┐
  ▼                                               ▼
[ Unsupervised Branch (clustering.py) ]   [ Supervised Branch (modeling.py) ]
  K-Means Clustering (K=4)                  Optuna Bayesian Optimization
  Dynamic Elbow Search                      Multi-Model Suite (RF, XGBoost,
  Visual Segmentation                                          LightGBM, CatBoost)
          │                                               │
          ▼                                               ▼
[ EDA & Cluster Plots (eda.py) ]          [ SHAP Explainability (explainability.py) ]
          │                                (Beeswarm Summary & Waterfall Plots)
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                   [ Artifacts Output Directory ]
           (eda_dashboard, elbow_method, segmentation,
            model_benchmark, shap_summary, shap_waterfall,
            model_performance, feature_importance)
```
