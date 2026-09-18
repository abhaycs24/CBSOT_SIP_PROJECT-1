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

## Churn Risk Scoring

The pipeline generates customer-level churn risk scores using the probability returned by the trained Random Forest model.

Customers are assigned a risk level based on their predicted churn probability:

| Churn Probability | Risk Level |
|---|---|
| >= 0.75 | High |
| 0.50 - 0.74 | Medium |
| < 0.50 | Low |

The generated risk table contains:

- Customer ID
- Churn Probability
- Risk Level

Customers are sorted by churn probability in descending order to help identify high-risk customers.

The pipeline also provides:

- High-risk customer identification
- A ranked customer risk list
- Churn probability for each customer
- Risk-level classification
- Churn risk distribution visualization

The implementation reuses the existing tuned Random Forest classifier and does not introduce a new prediction algorithm.
