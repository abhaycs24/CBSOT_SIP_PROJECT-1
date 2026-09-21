import argparse
import sys
import os
import pandas as pd

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from telco_churn.config import Config
from telco_churn.data import DataLoader
from telco_churn.validation import DataValidator
from telco_churn.preprocessing import DataPreprocessor
from telco_churn.feature_engineering import FeatureEngineer
from telco_churn.sampling import ClassBalancer
from telco_churn.eda import EDADashboard
from telco_churn.clustering import CustomerSegmentation
from telco_churn.modeling import ChurnPredictor
from telco_churn.evaluation import ModelEvaluator


def run_pipeline(config_path: str):
    """Executes the Telco Churn pipeline with Pydantic validation, feature engineering, and SMOTE-NC balancing."""
    print(f"Loading configuration from: {config_path}")
    cfg = Config.from_yaml(config_path)

    output_dir = cfg.get("output.artifacts_dir", "artifacts")

    # 1. Data Loading
    print("Step 1: Loading Dataset...")
    data_loader = DataLoader(raw_data_path=cfg.get("data.raw_data_path"))
    df_raw = data_loader.load_data(
        num_samples=cfg.get("data.num_samples", 100),
        random_state=cfg.get("data.random_state", 42),
        missing_charges_count=cfg.get("data.missing_charges_count", 5),
    )
    print(f"Dataset shape: {df_raw.shape}")

    # 2. Schema Validation (Pydantic)
    print("Step 2: Performing Data Quality Schema Validation (Pydantic)...")
    validator = DataValidator(enabled=cfg.get("validation.enabled", True))
    is_valid, valid_count, invalid_count = validator.validate(df_raw)
    print(f"Validation Status: {'PASS' if is_valid else 'FAIL'} (Valid records: {valid_count}, Invalid: {invalid_count})")

    # 3. Preprocessing & Feature Sanitization
    print("Step 3: Preprocessing & Data Cleaning...")
    preprocessor = DataPreprocessor(
        target_column=cfg.get("preprocessing.target_column", "Churn Label"),
        id_column=cfg.get("preprocessing.id_column", "customerID"),
    )
    df_clean = preprocessor.clean_data(df_raw)

    # 4. Feature Engineering Engine
    print("Step 4: Executing Feature Engineering Engine...")
    fe = FeatureEngineer(
        create_ratios=cfg.get("feature_engineering.create_ratios", True),
        create_contract_risk=cfg.get("feature_engineering.create_contract_risk", True),
    )
    df_engineered = fe.transform(df_clean, target_column=cfg.get("preprocessing.target_column", "Churn Label"))
    print(f"Engineered features added. Total columns: {df_engineered.shape[1]}")

    df_encoded, X, y, X_scaled = preprocessor.encode_and_scale(df_engineered)

    # 5. EDA Visual Dashboards
    print("Step 5: Generating Visual EDA Dashboard...")
    eda = EDADashboard()
    eda_path = eda.plot_dashboard(df_clean, output_dir=output_dir)
    print(f"EDA Dashboard saved to: {eda_path}")

    # 6. Customer Segmentation (K-Means)
    print("Step 6: Running Customer Segmentation (K-Means)...")
    segmentation = CustomerSegmentation(
        n_clusters=cfg.get("clustering.n_clusters", 4),
        elbow_k_max=cfg.get("clustering.elbow_k_max", 6),
        random_state=cfg.get("clustering.random_state", 42),
    )
    _, elbow_path = segmentation.plot_elbow_method(X_scaled, output_dir=output_dir)
    print(f"Elbow plot saved to: {elbow_path}")

    _, cluster_path = segmentation.fit_predict(df_clean, X_scaled, output_dir=output_dir)
    print(f"Cluster dashboard saved to: {cluster_path}")

    # 7. Supervised Split & SMOTE-NC Balancing
    print("Step 7: Supervised Data Split & SMOTE-NC Class Balancing...")
    predictor = ChurnPredictor(
        test_size=cfg.get("modeling.test_size", 0.2),
        random_state=cfg.get("modeling.random_state", 42),
        param_dist=cfg.get("modeling.param_dist"),
        n_iter=cfg.get("modeling.n_iter", 5),
        cv=cfg.get("modeling.cv", 3),
    )
    
    X_train, X_test, y_train, y_test = predictor.train_test_split(
        X, y, test_size=cfg.get("modeling.test_size", 0.2), random_state=cfg.get("modeling.random_state", 42), stratify=y
    )

    if cfg.get("sampling.use_smote_nc", True):
        balancer = ClassBalancer(random_state=cfg.get("sampling.random_state", 42))
        print(f"Pre-SMOTE Class Distribution: {dict(y_train.value_counts())}")
        X_train_res, y_train_res = balancer.balance_training_data(X_train, y_train)
        print(f"Post-SMOTE Class Distribution: {dict(pd.Series(y_train_res).value_counts())}")
    else:
        X_train_res, y_train_res = X_train, y_train

    # 8. Model Training & Hyperparameter Tuning
    print("Step 8: Training Random Forest with Hyperparameter Search...")
    predictor.best_model = predictor.fit_model(X_train_res, y_train_res)
    y_pred = predictor.best_model.predict(X_test)
    y_prob = predictor.best_model.predict_proba(X_test)[:, 1]

    # 9. Performance Evaluation & Artifact Exports
    print("\nStep 9: Evaluating Performance & Saving Plot Artifacts...")
    report = ModelEvaluator.generate_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    perf_path = ModelEvaluator.plot_performance_metrics(y_test, y_pred, y_prob, output_dir=output_dir)
    print(f"Model performance plots saved to: {perf_path}")

    imp_path = ModelEvaluator.plot_feature_importance(predictor.best_model, X.columns, output_dir=output_dir)
    print(f"Feature importance chart saved to: {imp_path}")

    print("\nPipeline execution complete successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Telco Customer Segmentation & Churn Prediction Modular Pipeline CLI"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    run_pipeline(args.config)


if __name__ == "__main__":
    main()
