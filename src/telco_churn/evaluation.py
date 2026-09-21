import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict


class ModelEvaluator:
    """Computes evaluation metrics and generates performance visual reports."""

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
        plt.title('Confusion Matrix')
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
        plt.title('ROC Curve Analysis')
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path

    @staticmethod
    def plot_feature_importance(
        model: RandomForestClassifier, X_columns: pd.Index, output_dir: str = "artifacts"
    ) -> str:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "feature_importance.png")

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig = plt.figure(figsize=(12, 7))
        sns.barplot(x=importances[indices][:10], y=X_columns[indices][:10], hue=X_columns[indices][:10], palette='mako', legend=False)
        plt.title('Top 10 Drivers of Customer Churn')
        plt.xlabel('Relative Importance Score')
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path
