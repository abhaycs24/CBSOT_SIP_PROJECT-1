import argparse
import sys
from telco_churn.config import Config
from telco_churn.data import DataLoader
from telco_churn.preprocessing import DataPreprocessor
from telco_churn.eda import EDADashboard
from telco_churn.clustering import CustomerSegmentation
from telco_churn.modeling import ChurnPredictor
from telco_churn.evaluation import ModelEvaluator


def run_pipeline(config_path: str):
    """Executes the complete Telco Customer Segmentation and Churn Prediction workflow."""
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

    # 2. Preprocessing
    print("Step 2: Preprocessing Data...")
    preprocessor = DataPreprocessor(
        target_column=cfg.get("preprocessing.target_column", "Churn Label"),
        id_column=cfg.get("preprocessing.id_column", "customerID"),
    )
    df_clean = preprocessor.clean_data(df_raw)
    df_encoded, X, y, X_scaled = preprocessor.encode_and_scale(df_clean)

    # 3. EDA Visualizations
    print("Step 3: Generating EDA Visual Dashboard...")
    eda = EDADashboard()
    eda_path = eda.plot_dashboard(df_clean, output_dir=output_dir)
    print(f"EDA Dashboard saved to: {eda_path}")

    # 4. Clustering & Customer Segmentation
    print("Step 4: Running Customer Segmentation (K-Means)...")
    segmentation = CustomerSegmentation(
        n_clusters=cfg.get("clustering.n_clusters", 4),
        elbow_k_max=cfg.get("clustering.elbow_k_max", 6),
        random_state=cfg.get("clustering.random_state", 42),
    )
    _, elbow_path = segmentation.plot_elbow_method(X_scaled, output_dir=output_dir)
    print(f"Elbow plot saved to: {elbow_path}")

    _, cluster_path = segmentation.fit_predict(df_clean, X_scaled, output_dir=output_dir)
    print(f"Cluster dashboard saved to: {cluster_path}")

    # 5. Supervised Model Training & Hyperparameter Tuning
    print("Step 5: Training Random Forest Classifier with Hyperparameter Tuning...")
    predictor = ChurnPredictor(
        test_size=cfg.get("modeling.test_size", 0.2),
        random_state=cfg.get("modeling.random_state", 42),
        param_dist=cfg.get("modeling.param_dist"),
        n_iter=cfg.get("modeling.n_iter", 5),
        cv=cfg.get("modeling.cv", 3),
    )
    X_train, X_test, y_train, y_test, y_pred, y_prob = predictor.train_and_tune(X, y)

    # 6. Evaluation & Artifact Exports
    print("\nStep 6: Evaluating Model Performance...")
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
