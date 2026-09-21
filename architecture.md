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
  Multi-Algorithm Clustering                Tuned Random Forest Classifier
  (K-Means, GMM, HDBSCAN)                   RandomizedSearchCV Hyperparameter Tuning
  Silhouette & Calinski Analysis            Feature Importance Extraction
  UMAP 2D Non-Linear Projections                          │
          │                                               │
          ▼                                               ▼
[ EDA & Cluster Plots (eda.py) ]            [ Evaluation Metrics (evaluation.py) ]
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                   [ Artifacts Output Directory ]
           (eda_dashboard, elbow_method, umap_clusters,
            confusion_matrix, roc_auc, feature_importance)
```
