# Telco Customer Segmentation & Churn Prediction Pipeline

This project builds an end-to-end data science workflow on the Telco subscriber dataset. The objective is twofold: first, to group customers based on behavioral attributes using unsupervised learning, and second, to predict potential churn using a tuned machine learning model.

## Key Insights & Objectives

1. **Exploratory Data Analysis**: Visualized continuous features like tenure months and monthly charges to spot outlier behaviors and class imbalances across subscriber metrics.
2. **Customer Segmentation**: Applied K-Means Clustering to partition the scaled feature space into 4 distinct customer personas, allowing targeted retention strategies.
3. **Supervised Modeling**: Implemented a Random Forest Classifier optimized via RandomizedSearchCV to accurately handle non-linear decision boundaries and identify primary churn drivers.

## Tech Stack Used
* **Environment**: Google Colab / Jupyter Notebooks
* **App/UI**: Streamlit
* **Data Core**: Python, Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn (StandardScaler, KMeans, RandomForestClassifier, RandomizedSearchCV), joblib for model persistence

## Pipeline Deliverables
* Dynamic Elbow Method plotting for optimal cluster selection.
* Behavior visualizations showing Customer Segments based on spend vs loyalty.
* Detailed Classification Report alongside Confusion Matrix and ROC-AUC Curve analysis to measure precise predictive power.
* Reusable training/export script (`train.py`) and an interactive Streamlit app (`app.py`) that serve the trained models without running the notebook.

## Project Structure

```text
CBSOT_SIP_PROJECT-1.ipynb    # Original analysis & modeling notebook
train.py                     # Trains the models and exports joblib artifacts
app.py                       # Streamlit application (loads the artifacts)
model_utils.py               # Shared inference helpers (risk tier, inputs, artifact loading)
artifacts/                   # Generated model artifacts (not committed)
tests/test_model_utils.py    # Unit + end-to-end artifact tests (pytest)
requirements.txt             # Pinned dependencies
```

## Run the Streamlit application

The app loads **pre-trained artifacts** from `artifacts/` — it never retrains the models during normal use.

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell / cmd):

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the model artifacts

```bash
python train.py
```

This trains the tuned Random Forest churn classifier and the K-Means (k=4) segmentation exactly like the notebook and writes the joblib artifacts to `artifacts/`:

| File | Contents |
| --- | --- |
| `artifacts/churn_pipeline.joblib` | Ordinal-encoder preprocessor + tuned Random Forest (`predict_proba`) |
| `artifacts/segment_pipeline.joblib` | StandardScaler + KMeans (k=4) |
| `artifacts/metadata.joblib` | Feature lists, risk thresholds, segment profiles, UI defaults, library versions |

By default `train.py` uses the notebook's synthetic sample dataset (seeded with 42 for reproducibility). To train on a real export with the same column layout (`customerID, Gender, Tenure Months, Monthly Charges, Total Charges, Contract, Churn Label`), run:

```bash
python train.py --data path/to/telco.csv
```

### 4. Launch the app

```bash
streamlit run app.py
```

### 5. Using the sidebar

1. Enter the customer's **Gender**, **Tenure (months)**, **Monthly charges ($)**, **Total charges ($)** and **Contract type** — these are exactly the five features the models were trained on.
2. Press **Predict churn**.
3. The main panel shows the model-estimated churn probability, the risk tier, the assigned K-Means segment and suggested retention actions.

### Churn risk tiers

The probability comes from the Random Forest's `predict_proba()` for the churn class (1). Tiers use the thresholds stored in `artifacts/metadata.joblib`:

| Tier | Probability | Typical actions |
| --- | --- | --- |
| **Low** | `< 0.33` | Loyalty benefits, referrals, cross-sell, satisfaction monitoring |
| **Medium** | `0.33 – 0.66` | Engagement campaigns, plan education, payment-method assistance |
| **High** | `>= 0.66` | Proactive outreach, service review, targeted discounts, contract-renewal incentives, technical-support follow-up |

The displayed probability is a **model estimate** based on historical patterns, not a guarantee of future behaviour.

### Customer segments

The K-Means cluster labels (0–3) are arbitrary; `train.py` profiles every cluster on the training data (average tenure, average monthly/total charges, most common contract, churn rate) and derives a business name from the tenure/spend quadrants, for example "High-value loyal customers" or "New low-spend customers". Those names, descriptions and per-segment recommendations are stored in `artifacts/metadata.joblib` and shown in the app together with a combined, risk-aware retention suggestion. If a cluster has no stored profile, the app shows a neutral label ("Segment N") and a generic recommendation instead of guessing.

### Retraining / regenerating artifacts

Regenerate the artifacts whenever the training data, preprocessing, feature definitions or model configuration change:

```bash
python train.py            # same defaults as above
python train.py --data telco.csv --artifacts-dir artifacts
```

The app loads the artifacts once per process (Streamlit resource cache); restart the app after regenerating them.

### Testing

```bash
pytest -q
```

The suite covers risk-tier boundaries, positive-class probability extraction, input-frame building, segment mapping, artifact creation/loading, corrupt/missing artifact handling and an end-to-end sample prediction.

### Troubleshooting

* **"Model artifacts are missing"** — run `python train.py` first, then restart `streamlit run app.py`.
* **"Model artifacts could not be loaded"** — the joblib files are corrupt or were created with an incompatible library version; delete `artifacts/` and regenerate them with `python train.py`.
* **Prediction fails in the app** — regenerate the artifacts so they match your installed scikit-learn/joblib versions (see below), then restart the app.
* **Wrong window after regenerating artifacts** — the app caches artifacts per process; stop and relaunch `streamlit run app.py`.

### Artifact version compatibility

joblib artifacts are version-sensitive. The committed artifacts (if any) were generated with:

* Python **3.12.4**, scikit-learn **1.7.1**, joblib **1.5.1**, pandas **2.3.1**, numpy **2.3.2**

The pinned versions in `requirements.txt` match these. The previous pins (`pandas 2.0.3`, `numpy 1.24.3`, `scikit-learn 1.2.2`, `matplotlib 3.7.1`, `seaborn 0.12.2`) have no wheels for Python ≥ 3.12 and were updated accordingly; if you must stay on those older versions, regenerate the artifacts with `python train.py` inside that environment instead of reusing `artifacts/`.

The `artifacts/` directory is intentionally **not committed** (see `.gitignore`) — always generate it locally with `python train.py`. Customer data should never be committed to this repository.


