# Telco Customer Segmentation & Churn Prediction Pipeline

This project provides an end-to-end data science workflow on the Telco subscriber dataset. The objective is twofold: first, to group customers based on behavioral attributes using unsupervised learning, and second, to serialize trained ML model artifacts, serve real-time predictions via a **FastAPI** REST microservice, and provide an interactive **Streamlit** Web Intelligence Dashboard.

## Key Features & Highlights

1. **Model Artifact Serialization**: Automated persistence of trained models and fitted preprocessor pipelines (`models/artifacts/model.joblib`).
2. **FastAPI REST Prediction Microservice**: High-performance API (`src/telco_churn/api.py`) exposing endpoints:
   - `GET /health`: Microservice status check.
   - `POST /predict`: Real-time single customer churn risk scoring.
   - `POST /predict-batch`: Bulk CSV batch prediction payload processing.
3. **Interactive Streamlit Web App**: Real-time interactive UI (`app.py`) allowing business users to input subscriber attributes, view churn probabilities, and inspect visual artifact dashboards.
4. **Docker MLOps Containerization**: Multi-stage `Dockerfile` for containerized deployment across cloud infrastructure.
5. **Modular Package Architecture**: Object-oriented Python package (`src/telco_churn/`) separating data ingestion, preprocessing, EDA, clustering, modeling, serialization, API serving, and evaluation.

## Tech Stack Used
* **CLI Engine**: Python, PyYAML, Argparse
* **REST Microservice**: FastAPI, Uvicorn, Pydantic
* **Web UI**: Streamlit
* **ML & MLOps**: Scikit-Learn, Joblib, Docker
* **Data Core**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn

## Directory Layout
```text
CBSOT_SIP_PROJECT-1/
├── configs/
│   └── default.yaml         # YAML Configuration settings
├── models/
│   └── artifacts/           # Serialized Joblib models & preprocessors
├── src/
│   └── telco_churn/         # Modular Package Source
│       ├── __init__.py
│       ├── config.py        # YAML configuration parser
│       ├── data.py          # Data ingestion and synthetic generator
│       ├── preprocessing.py # Preprocessing & feature scaling
│       ├── eda.py           # Exploratory data visualization
│       ├── clustering.py    # K-Means clustering & Elbow method
│       ├── modeling.py      # Random Forest model & hyperparameter tuning
│       ├── evaluation.py    # Metrics report & plot generation
│       ├── serialization.py # Model & preprocessor Joblib serializer
│       ├── api.py           # FastAPI REST prediction service
│       └── cli.py           # Command Line Interface runner
├── app.py                   # Streamlit Web Application
├── Dockerfile               # Container deployment spec
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

### 2. Run the Full Pipeline & Train Model
Execute the CLI pipeline to train and serialize model artifacts:
```bash
python src/telco_churn/cli.py --config configs/default.yaml
```

### 3. Launch the FastAPI REST Service
Start the real-time prediction microservice:
```bash
uvicorn src.telco_churn.api:app --reload --port 8000
```
Swagger API docs available at: `http://localhost:8000/docs`

### 4. Launch the Streamlit Web Application
Start the interactive Streamlit UI:
```bash
streamlit run app.py
```
App interface available at: `http://localhost:8501`

### 5. Run via Docker Container
Build and launch the containerized MLOps environment:
```bash
docker build -t telco-churn-app .
docker run -p 8000:8000 -p 8501:8501 telco-churn-app
```

## Pipeline Deliverables & Artifacts
All generated visualizations are saved automatically in the `artifacts/` folder:
* `eda_dashboard.png`: Feature distributions and correlation heatmaps.
* `elbow_method.png`: Dynamic Elbow Method plotting for optimal cluster selection.
* `customer_segmentation.png`: Visualizing customer personas based on spend vs. loyalty.
* `model_performance.png`: Confusion Matrix and ROC-AUC Curve analysis.
* `feature_importance.png`: Top 10 primary drivers of customer churn.
