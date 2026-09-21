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
        [ Pydantic Validation (validation.py) ]
      (Data Types, Range Bounds & Non-Null Checks)
                          │
                          ▼
       [ Preprocessing (preprocessing.py) ]
   (String Sanitization, Imputation, Scaling)
                          │
                          ▼
   [ Feature Engineering (feature_engineering.py) ]
  (Tenure/Charge Ratios, Contract Risk Index)
                          │
                          ▼
  ┌───────────────────────┴───────────────────────┐
  ▼                                               ▼
[ Unsupervised Branch (clustering.py) ]   [ Supervised Branch (sampling.py + modeling.py) ]
  K-Means Clustering (K=4)                  SMOTE-NC Class Imbalance Resampling
  Dynamic Elbow Search                      Tuned Random Forest Classifier
  Visual Segmentation                       RandomizedSearchCV Hyperparameter Tuning
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
