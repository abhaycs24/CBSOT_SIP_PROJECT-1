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

## Dataset Setup
1. Download the Telco Customer Churn dataset and place the CSV file in the project root directory.
2. The notebook expects the dataset file to be named:
3. `WA_Fn-UseC_-Telco-Customer-Churn.csv`
4. The dataset is used for preprocessing, exploratory data analysis, customer segmentation, and churn prediction.

## Pipeline Deliverables
* Dynamic Elbow Method plotting for optimal cluster selection.
* Behavior visualizations showing Customer Segments based on spend vs loyalty.
* Detailed Classification Report alongside Confusion Matrix and ROC-AUC Curve analysis to measure precise predictive power.
