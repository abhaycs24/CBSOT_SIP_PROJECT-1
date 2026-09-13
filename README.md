# Telco Customer Segmentation & Churn Prediction Pipeline

This project builds an end-to-end data science workflow on the Telco subscriber dataset. The objective is twofold: first, to group customers based on behavioral attributes using unsupervised learning, and second, to predict potential churn using a tuned machine learning model.

## Key Insights & Objectives

1. **Exploratory Data Analysis**: Visualized continuous features like tenure months and monthly charges to spot outlier behaviors and class imbalances across subscriber metrics.
2. **Customer Segmentation**: Applied K-Means Clustering to partition the scaled feature space into 4 distinct customer personas, allowing targeted retention strategies.
3. **Supervised Modeling**: Implemented a Random Forest Classifier optimized via RandomizedSearchCV to accurately handle non-linear decision boundaries and identify primary churn drivers.

## Tech Stack Used
* **Environment**: Google Colab / Jupyter Notebooks
* **Data Core**: Python, Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-Learn (StandardScaler, KMeans, RandomForestClassifier, RandomizedSearchCV)

## Pipeline Deliverables
* Dynamic Elbow Method plotting for optimal cluster selection.
* Behavior visualizations showing Customer Segments based on spend vs loyalty.
* Detailed Classification Report alongside Confusion Matrix and ROC-AUC Curve analysis to measure precise predictive power.
## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/abeerahateeq/CBSOT_SIP_PROJECT-1-.git
cd CBSOT_SIP_PROJECT-1-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the Notebook

Open `CBSOT_SIP_PROJECT-1.ipynb` using **Google Colab** or **Jupyter Notebook**.

### 4. Run the Notebook

Run the notebook cells from top to bottom.

The notebook performs:

* Data preprocessing
* Exploratory data analysis
* Customer segmentation using K-Means
* Churn prediction using Random Forest
* Model evaluation
* Feature importance analysis

## Project Structure

```text
CBSOT_SIP_PROJECT-1-/
│
├── CBSOT_SIP_PROJECT-1.ipynb
├── README.md
├── architecture.md
└── requirements.txt
```

### File Description

* `CBSOT_SIP_PROJECT-1.ipynb` — Main Jupyter/Colab notebook containing the data analysis and machine learning pipeline.
* `README.md` — Project documentation and usage instructions.
* `architecture.md` — Overview of the project's workflow and architecture.
* `requirements.txt` — Python dependencies required to run the project.

