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

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/abhaycs24/CBSOT_SIP_PROJECT-1-.git
cd CBSOT_SIP_PROJECT-1-

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dataset

This project uses the **IBM Telco Customer Churn** dataset.

Download it from Kaggle:  
🔗 [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

After downloading, place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project root directory.

### Running the Notebook

```bash
# Launch Jupyter Notebook
jupyter notebook CBSOT_SIP_PROJECT-1.ipynb
```

Or open and run it directly in [Google Colab](https://colab.research.google.com/) by uploading the `.ipynb` file.
