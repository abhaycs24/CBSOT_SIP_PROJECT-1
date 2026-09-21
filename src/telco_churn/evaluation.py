import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from typing import Tuple, Dict, Any


class ModelEvaluator:
    """Computes evaluation metrics and generates multi-model performance visual reports."""

    @staticmethod
    def generate_report(y_test: pd.Series, y_pred: np.ndarray) -> str:
        return classification_report(y_test, y_pred)

    @staticmethod
    def plot_performance_metrics(
        y_test: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray, output_dir: str = "artifacts"
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "model_performance.png")

        fig = plt.figure(figsize=(14, 6))

        # Confusion Matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Best Model)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # ROC Curve
        plt.subplot(1, 2, 2)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_score = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Analysis (Best Model)')
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path

    @staticmethod
    def plot_model_benchmark(
        benchmark_models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        benchmark_df: pd.DataFrame,
        output_dir: str = "artifacts",
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "model_benchmark.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Overlay ROC Curves across all benchmarked models
        for name, model in benchmark_models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            ax1.plot(fpr, tpr, lw=2, label=f"{name.upper()} (AUC = {auc_val:.2f})")

        ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Comparative Model ROC-AUC Overlay')
        ax1.legend(loc="lower right")

        # 2. Metric comparison bar chart
        benchmark_reset = benchmark_df.reset_index().rename(columns={"index": "Model"})
        melted_df = pd.melt(benchmark_reset, id_vars=["Model"], value_vars=["ROC-AUC", "Accuracy", "F1-Score"])
        sns.barplot(data=melted_df, x="Model", y="value", hue="variable", ax=ax2, palette="viridis")
        ax2.set_title("Multi-Model Benchmark Performance Comparison")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path

    @staticmethod
    def plot_feature_importance(
        model: Any, X_columns: pd.Index, output_dir: str = "artifacts"
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "feature_importance.png")

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            return save_path

        indices = np.argsort(importances)[::-1]

        fig = plt.figure(figsize=(12, 7))
        sns.barplot(
            x=importances[indices][:10],
            y=X_columns[indices][:10],
            hue=X_columns[indices][:10],
            palette='mako',
            legend=False,
        )
        plt.title('Top 10 Drivers of Customer Churn')
        plt.xlabel('Relative Importance Score')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path
