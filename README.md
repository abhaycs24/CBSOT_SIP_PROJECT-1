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

## Dataset

This pipeline is designed for the **Telco Customer Churn** dataset (IBM sample dataset, commonly distributed as `WA_Fn-UseC_-Telco-Customer-Churn.csv`), available on Kaggle:
[https://www.kaggle.com/datasets/blastchar/telco-customer-churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the link above.
2. Place it in the **same folder as the notebook**.
3. Just run the notebook — it automatically detects the file and loads it. No code changes needed.

> **Note:** If the CSV isn't found, the notebook automatically falls back to a small synthetic placeholder dataset so it can still run end-to-end (useful for a quick demo or if you just want to see the pipeline work before downloading the real data).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abhaycs24/CBSOT_SIP_PROJECT-1-.git
   cd CBSOT_SIP_PROJECT-1-
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Requires **Python 3.9+** (tested on Python 3.12).

## Execution

1. Launch Jupyter Notebook or open the notebook in Google Colab:
   ```bash
   jupyter notebook CBSOT_SIP_PROJECT-1.ipynb
   ```
2. Ensure the dataset CSV is in the same directory as the notebook (see [Dataset](#dataset) above).
3. Run all cells sequentially (`Cell > Run All`). The notebook will:
   - Display EDA visualizations (churn ratio, correlation heatmap, tenure vs. charges, contract impact)
   - Plot the Elbow Method curve and generate K-Means customer segments
   - Train and tune the Random Forest classifier
   - Output the classification report, confusion matrix, ROC-AUC curve, and top churn drivers
