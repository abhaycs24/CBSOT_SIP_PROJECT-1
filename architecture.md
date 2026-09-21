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
  K-Means Clustering (K=4)                  Tuned Random Forest Classifier
  Dynamic Elbow Search                      RandomizedSearchCV Optimization
  Visual Segmentation                       Feature Importance Extraction
          │                                               │
          ▼                                               ▼
[ EDA & Cluster Plots (eda.py) ]            [ Evaluation Metrics (evaluation.py) ]
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  ▼
                   [ Artifacts Output Directory ]
           (eda_dashboard, elbow_method, segmentation,
            confusion_matrix, roc_auc, feature_importance)
```
