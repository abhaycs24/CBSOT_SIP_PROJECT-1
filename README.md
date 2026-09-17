# 📡 Telco Customer Segmentation & Churn Prediction System

An end-to-end, production-ready Machine Learning system built on the canonical IBM Telco subscriber dataset. This platform pairs **Unsupervised Customer Behavioral Segmentation (K-Means)** with **Supervised Churn Risk Prediction (Tuned Random Forest)**, served through a multi-view interactive **Streamlit analytics dashboard**.

---

## 📌 Project Overview

Customer churn poses a significant threat to subscription and telecommunication businesses. This project addresses churn prevention through a dual-perspective machine learning pipeline:
1. **Unsupervised Customer Segmentation (K-Means):** Identifies distinct subscriber personas based on customer lifecycle tenure and spending patterns to inform personalized marketing and retention programs.
2. **Supervised Churn Prediction (Random Forest + RandomizedSearchCV):** Accurately flags individual subscribers at risk of cancellation, calculates calibrated churn probabilities, and isolates key statistical churn drivers.
3. **Interactive Analytics Web Application (Streamlit):** Translates raw machine learning pipelines into actionable business dashboards with live real-time customer scoring, batch CSV uploads, and model explainability.

---

## 🏛️ System Architecture

```text
               ┌──────────────────────────────────────────────┐
               │    Raw Telco Dataset (7,043 Subscribers)     │
               │        data/Telco-Customer-Churn.csv         │
               └──────────────────────┬───────────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │   Data Cleaning & Schema Harmonization       │
               │   (TotalCharges imputation, type casting,    │
               │    whitespace stripping, ID isolation)       │
               └──────────────────────┬───────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
       ┌───────────────────────────┐     ┌───────────────────────────┐
       │   Unsupervised Branch     │     │    Supervised Branch      │
       │   (Customer Segmentation) │     │    (Churn Prediction)     │
       ├───────────────────────────┤     ├───────────────────────────┤
       │ • Feature Extraction:     │     │ • Stratified Split (80/20)│
       │   tenure, MonthlyCharges, │     │ • ColumnTransformer:      │
       │   TotalCharges            │     │   - StandardScaler        │
       │ • StandardScaler          │     │   - OneHotEncoder(ignore) │
       │ • Elbow Method (K=1..10)  │     │ • RandomForestClassifier  │
       │ • K-Means (K=4)           │     │ • RandomizedSearchCV      │
       │ • Empirical Profiles      │     │   (3-Fold CV, ROC-AUC)    │
       └─────────────┬─────────────┘     └─────────────┬─────────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
               ┌──────────────────────────────────────────────┐
               │       Model Serialization via Joblib         │
               │   (models/*.joblib: Preprocessor, K-Means,   │
               │    Tuned RF, Test Metrics, Importances)      │
               └──────────────────────┬───────────────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
       ┌───────────────────────────┐     ┌───────────────────────────┐
       │    Streamlit Dashboard    │     │    Pytest Verification    │
       │  (Interactive Analytics & │     │ (Preprocessing, Pipeline, │
       │   Live Model Inference)   │     │  Inference, Schema tests) │
       └───────────────────────────┘     └───────────────────────────┘
```

---

## 📊 Dataset & Preprocessing

### Dataset Schema
* **Source:** IBM Telco Customer Churn dataset (7,043 subscriber records, 21 columns).
* **Target:** `Churn` (Yes = 1,869 [26.5%], No = 5,174 [73.5%]).
* **Numerical Features:** `tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`.
* **Categorical Features:** `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`.

### Data Cleaning & Leakage Prevention
1. **Schema Harmonization:** Automatically detects and aligns column name variants (e.g., `Tenure Months` $\rightarrow$ `tenure`, `Monthly Charges` $\rightarrow$ `MonthlyCharges`, `Churn Label` $\rightarrow$ `Churn`).
2. **Missing Value Sanitization:** Converts whitespace strings (`' '`) in `TotalCharges` into `NaN`. For subscribers with `tenure == 0`, `TotalCharges` is assigned `$0.00`. Any remaining missing total charges are imputed using the median ($1,397.48).
3. **Leakage-Free Preprocessing:** Features are split via stratified 80/20 partition before any modeling. A Scikit-Learn `ColumnTransformer` (`StandardScaler` for numeric, `OneHotEncoder` with `handle_unknown='ignore'` for categorical) is fitted **strictly on the training set** and serialized to eliminate training-serving skew.

---

## 👥 Customer Segmentation (K-Means)

Clustering is applied to scaled behavioral spending features: `tenure`, `MonthlyCharges`, and `TotalCharges`.

### Elbow Method Analysis
The Within-Cluster Sum of Squares (Inertia) was calculated across $K=1 \dots 10$. The curve reveals an elbow at **$K=4$**, yielding four operationally distinct segments with stark differences in retention behavior:

### Empirical Cluster Statistics (Actual Dataset)

| Cluster ID | Persona Name | Subscribers | Share (%) | Avg Tenure | Avg Monthly ($) | Avg Total ($) | Actual Churn Rate (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **New Budget Starters** | 1,703 | 24.2% | 10.2 mos | $31.78 | $355.67 | **24.7%** |
| **1** | **High-Value Loyal Champions** | 1,904 | 27.0% | 59.5 mos | $93.31 | $5,648.57 | **15.4%** |
| **2** | **New High-Spend / At-Risk** | 2,276 | 32.3% | 15.4 mos | $80.78 | $1,348.65 | **48.2%** |
| **3** | **Established Budget Retained** | 1,160 | 16.5% | 53.6 mos | $34.91 | $1,940.66 | **5.0%** |

> **Key Insight:** While Cluster 3 (long tenure, budget spend) experiences an exceptional **5.0% churn rate**, Cluster 2 (early lifecycle, premium monthly charges) exhibits a critical **48.2% churn rate**, providing a clear target for proactive onboarding interventions.

---

## 🤖 Churn Prediction Model (Random Forest)

### Training & Hyperparameter Tuning
* **Algorithm:** `RandomForestClassifier`
* **Optimization:** `RandomizedSearchCV` (3-Fold Stratified Cross-Validation, 10 iterations, scoring on `roc_auc`).
* **Optimal Hyperparameters:**
  ```python
  {
      'n_estimators': 50,
      'max_depth': 8,
      'min_samples_split': 10,
      'min_samples_leaf': 4,
      'max_features': 'log2',
      'class_weight': None
  }
  ```
* **Best Cross-Validation ROC-AUC:** `0.8482`

### Genuine Held-Out Test Set Results (1,409 Subscribers)

| Metric | Score | Performance Note |
| :--- | :---: | :--- |
| **Accuracy** | **80.06%** | Overall correct test classifications |
| **ROC-AUC** | **0.8430** | Strong discriminative ability across decision thresholds |
| **Precision (Churn)** | **66.55%** | Minimizes false alarms on loyal subscribers |
| **Recall (Churn)** | **50.00%** | Catches 1 out of 2 actual churners at default 0.50 threshold |
| **F1-Score (Churn)** | **0.5710** | Balanced harmonic mean of precision and recall |

#### Test Set Confusion Matrix (Threshold = 0.50)
* **True Negatives (TN):** 941 (66.8%)
* **False Positives (FP):** 94 (6.7%)
* **False Negatives (FN):** 187 (13.3%)
* **True Positives (TP):** 187 (13.3%)

---

## 🔍 Top Predictive Drivers (Feature Importance)

The top statistical predictors identified by the tuned Random Forest model:

1. **Tenure (Months)** (Importance: `0.1413`)
2. **Total Charges ($)** (Importance: `0.1040`)
3. **Contract: Month-to-month** (Importance: `0.0921`)
4. **Contract: Two year** (Importance: `0.0582`)
5. **TechSupport: No** (Importance: `0.0562`)
6. **OnlineSecurity: No** (Importance: `0.0558`)
7. **InternetService: Fiber optic** (Importance: `0.0469`)
8. **PaymentMethod: Electronic check** (Importance: `0.0435`)

> ⚠️ **Predictive Association vs. Causation:**
> Feature importance scores reflect statistical association in tree split decisions, not causal levers. For example, while Month-to-month contracts strongly correlate with churn, forcing long-term commitments without addressing root customer dissatisfaction may not directly reduce customer attrition.

---

## 💻 Streamlit Web Application Interface

The Streamlit dashboard (`app.py`) provides five dedicated views:

1. **Dashboard:** High-level executive KPIs, subscriber base overview, interactive churn ratio donut chart, and empirical customer segment breakdown.
2. **Customer Segments:** Elbow curve visualization (justifying $K=4$), interactive 2D scatter plot (Tenure vs. Monthly Charges), empirical statistics table, and strategic persona action plans.
3. **Churn Prediction:** Interactive customer profile input form running live inference against the trained pipelines, returning churn probability, decision threshold comparison, and assigned customer persona.
4. **Model Performance:** Test set metrics cards, interactive Confusion Matrix, ROC curve with AUC benchmark, ranked feature importance chart, and metric education guide.
5. **Data Explorer:** Searchable dataset table, missing values audit, interactive continuous metric distributions (histograms + boxplots), categorical churn rate cross-tabs, and a batch CSV scoring tool.

---

## 🚀 Installation & Execution

### 1. Environment Setup
Clone the repository and install dependencies:
```bash
git clone <repository_url>
cd SIP
pip install -r requirements.txt
```

### 2. Run Model Training Pipeline
Train the K-Means segmentation model, optimize the Random Forest classifier, and serialize all artifacts to `models/`:
```bash
python train.py
```

### 3. Run Automated Tests
Execute the comprehensive Pytest verification suite:
```bash
python -m pytest tests/ -v
```

### 4. Launch the Streamlit Dashboard
Launch the interactive web application locally:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
SIP/
├── app.py                      # Streamlit interactive analytics dashboard
├── train.py                    # End-to-end training & evaluation pipeline
├── requirements.txt            # Pinned dependencies
├── architecture.md             # Detailed architecture specification
├── README.md                   # Comprehensive project documentation
├── data/
│   └── Telco-Customer-Churn.csv # Verified IBM Telco dataset (7,043 records)
├── models/                     # Serialized joblib artifacts
│   ├── churn_rf_model.joblib   # Tuned Random Forest Classifier
│   ├── preprocessor.joblib     # Fitted ColumnTransformer pipeline
│   ├── kmeans_model.joblib     # Fitted K-Means model (K=4)
│   ├── kmeans_scaler.joblib    # StandardScaler for segmentation features
│   ├── cluster_profiles.joblib # Empirical summary statistics per cluster
│   ├── elbow_data.joblib       # WCSS inertias for K=1..10
│   ├── model_metrics.joblib    # Held-out test set accuracy, ROC-AUC, CM
│   └── feature_importances.joblib # Ranked Gini importance table
├── notebooks/
│   └── CBSOT_SIP_PROJECT-1.ipynb # Preserved exploratory notebook
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py  # Data cleaning, schema harmonization, pipeline factory
│   ├── segmentation.py        # K-Means, Elbow method, empirical cluster persona engine
│   ├── churn_model.py         # Supervised train/test split, RF & RandomizedSearchCV
│   ├── evaluation.py          # Metric calculations, confusion matrix, feature ranking
│   ├── prediction.py          # Inference engine for live single and batch predictions
│   └── visualization.py       # Plotly-powered responsive charts for dashboard
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py  # Unit tests for cleaning and ColumnTransformer
    ├── test_segmentation.py   # Unit tests for Elbow curve and K-Means logic
    ├── test_model.py          # Unit tests for saved model artifacts & metric validity
    └── test_prediction.py     # Unit tests for live single and batch inference
```

---

## 🔬 Limitations & Future Improvements

* **Class Imbalance:** Churn accounts for 26.5% of records. While balanced class weights and threshold tuning provide strong discrimination (0.843 AUC), incorporating SMOTE or cost-sensitive learning could further boost minority recall.
* **Customer Lifetime Value (CLTV):** Future releases can integrate survival analysis (Kaplan-Meier, Cox Proportional Hazards) to model expected time-to-churn and estimate remaining customer lifetime value.
* **Causal Inference:** Supplement predictive Gini importances with Uplift Modeling (e.g., CausalML, DoWhy) to determine which customers actively benefit from retention offers vs. those who would renew organically.

---

## 📜 License & Acknowledgments
Dataset provided by IBM Business Analytics community. Built with Python, Scikit-Learn, Streamlit, and Plotly.
