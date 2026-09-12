# Telco Customer Segmentation & Churn Prediction Pipeline

## Overview

This project builds an end-to-end data science workflow on the Telco Customer dataset. It addresses two complementary objectives:

- **Customer Segmentation:** Identify distinct customer groups based on behavioral and account characteristics using unsupervised learning.
- **Churn Prediction:** Build a supervised machine learning model to predict customers who are more likely to churn.

The goal is to combine customer segmentation and churn prediction to generate insights that can support targeted customer retention strategies.

## Key Insights & Objectives

### 1. Exploratory Data Analysis

The dataset is explored using statistical analysis and visualizations to understand customer behavior, identify patterns, and detect potential outliers.

Key features analyzed include:

- Customer tenure
- Monthly charges
- Customer spending behavior
- Churn distribution
- Other relevant customer and account attributes

Visualizations are created using Matplotlib and Seaborn, including:

- Continuous-feature analysis of tenure in months and monthly charges
- Outlier detection and behavioral pattern exploration
- Churn distribution and class-imbalance analysis across subscriber metrics

### 2. Customer Segmentation

K-Means Clustering is used as an unsupervised learning technique to group customers with similar characteristics.

The workflow includes:

- Selecting relevant customer features
- Scaling numerical features using `StandardScaler`
- Evaluating different numbers of clusters using the Elbow Method
- Applying K-Means clustering
- Analyzing the characteristics of the resulting customer segments
- Visualizing customer segments based on behavioral attributes

The current implementation uses **4 customer segments** to represent distinct customer personas. The notebook includes a dynamic Elbow Method plot to compare candidate cluster counts and support selection of the optimal value of `K`.

Customer behavior is visualized using spend versus loyalty indicators, such as monthly charges and tenure, so the resulting segments can be interpreted as actionable customer personas.

These segments can potentially be used to design more targeted engagement and retention strategies.

### 3. Churn Prediction

A Random Forest Classifier is used to predict customer churn.

Random Forest is suitable for this task because it can model non-linear relationships between customer characteristics and churn behavior while combining predictions from multiple decision trees.

The model development workflow includes:

- Preparing the data for supervised learning
- Splitting the data into training and testing sets
- Training a Random Forest classifier
- Hyperparameter optimization using `RandomizedSearchCV`
- Generating churn predictions
- Evaluating model performance

The tuned model is designed to identify non-linear relationships between customer attributes and churn risk, helping prioritize retention efforts.

## Key Insights & Business Objectives

- **Behavioral understanding:** Tenure and monthly charges reveal differences in loyalty and spending behavior.
- **Customer personas:** K-Means clustering groups subscribers into four segments that can support differentiated engagement strategies.
- **Retention prioritization:** A tuned Random Forest highlights customers who may be at higher risk of churn.
- **Evidence-based evaluation:** Classification metrics and diagnostic plots provide a balanced view of predictive performance.

## Machine Learning Pipeline

```text
Telco Customer Dataset
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Preparation
        ↓
Feature Scaling
        ↓
K-Means Customer Segmentation
        ↓
Customer Segment Analysis
        ↓
Random Forest Churn Prediction
        ↓
Hyperparameter Optimization
        ↓
Model Evaluation
        ↓
Customer & Business Insights
```

## Tech Stack

| Category | Technologies |
| --- | --- |
| Language | Python |
| Development Environment | Google Colab / Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Feature Scaling | Scikit-learn `StandardScaler` |
| Clustering | Scikit-learn `KMeans` |
| Classification | Scikit-learn `RandomForestClassifier` |
| Hyperparameter Optimization | Scikit-learn `RandomizedSearchCV` |
| Model Evaluation | Classification Report, Confusion Matrix, ROC-AUC |

## Project Deliverables

The project provides the following analysis and outputs:

- Exploratory data analysis of customer characteristics
- Visualizations of customer behavior and churn patterns
- Dynamic Elbow Method plotting for selecting the number of clusters
- K-Means customer segmentation
- Customer segment visualizations, including spend-versus-loyalty analysis
- Random Forest churn prediction
- Hyperparameter optimization using RandomizedSearchCV
- Detailed Classification Report
- Confusion Matrix
- ROC-AUC curve and analysis
- Insights that can support customer retention strategies

## Model Evaluation

The churn prediction model is evaluated using multiple classification metrics rather than relying on a single performance measure.

### Classification Report

The classification report provides metrics such as:

- **Precision:** Proportion of predicted churn cases that are actually churn cases.
- **Recall:** Proportion of actual churn cases correctly identified by the model.
- **F1-score:** Harmonic mean of precision and recall.
- **Support:** Number of samples belonging to each class.

### Confusion Matrix

The confusion matrix provides a breakdown of:

- True Positives
- True Negatives
- False Positives
- False Negatives

This helps identify the types of prediction errors made by the model.

### ROC-AUC

The ROC-AUC score is used to evaluate the model's ability to distinguish between customers who churn and customers who do not churn.

Using multiple evaluation methods provides a more complete understanding of model performance.

## Project Structure

```text
.
├── CBSOT_SIP_PROJECT-1.ipynb
├── README.md
├── architecture.md
└── requirements.txt
```

### File Descriptions

- `CBSOT_SIP_PROJECT-1.ipynb` — Main notebook containing data analysis, preprocessing, clustering, model training, and evaluation.
- `README.md` — Project documentation and usage information.
- `architecture.md` — High-level project architecture and workflow.
- `requirements.txt` — Python dependencies required to run the project.

## Getting Started

### Prerequisites

Make sure you have Python installed on your system.

You can verify your installation with:

```bash
python --version
```


Git is also recommended for cloning the repository.

### Installation

Clone the repository:

```bash
git clone https://github.com/abhaycs24/CBSOT_SIP_PROJECT-1-.git
```


Navigate to the project directory:

```bash
cd CBSOT_SIP_PROJECT-1-
```


Create a virtual environment:

```bash
python -m venv .venv
```


Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```


On macOS/Linux:

```bash
source .venv/bin/activate
```


Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Project

The main analysis is provided as a Jupyter Notebook:

Open `CBSOT_SIP_PROJECT-1.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.

Run the notebook cells sequentially to reproduce the data preprocessing, exploratory analysis, clustering, churn prediction, and model evaluation workflow.

## Future Improvements

Potential future improvements include:

- Comparing K-Means with other clustering algorithms
- Experimenting with additional classification models
- Improving feature engineering
- Performing more extensive hyperparameter optimization
- Adding automated tests
- Adding model explainability techniques
- Improving data validation and preprocessing
- Creating a reproducible training pipeline
- Adding automated model evaluation
- Deploying the trained model through an API or web application

## Contributing

Contributions are welcome!

If you would like to contribute:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make and test your changes.
4. Commit your changes with a clear commit message.
5. Push your branch to your fork.
6. Open a pull request describing your changes.

For documentation improvements, Python improvements, data-processing enhancements, testing, or machine learning experiments, please provide a clear explanation of the motivation and expected impact of the change.

## License

Please refer to the repository for the project's current licensing information.