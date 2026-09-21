import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from typing import Tuple, Any


class SHAPExplainer:
    """Provides SHAP model interpretability (Beeswarm, Feature Importance, Waterfall plots)."""

    def __init__(self, model: Any):
        self.model = model
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = shap.Explainer(model)

    def explain(self, X_test: pd.DataFrame, output_dir: str = "artifacts") -> Tuple[np.ndarray, str, str]:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "shap_summary.png")
        waterfall_path = os.path.join(output_dir, "shap_waterfall.png")

        shap_values = self.explainer(X_test)

        # Handle multi-output (e.g. classification 2D output array)
        if len(shap_values.shape) == 3:
            # Binary classification (samples, features, classes) -> take positive class (1)
            target_shap_values = shap_values[:, :, 1]
        else:
            target_shap_values = shap_values

        # 1. SHAP Summary Beeswarm Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(target_shap_values, X_test, show=False)
        plt.title("SHAP Global Feature Impact (Beeswarm Summary)", fontsize=14)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. SHAP Waterfall Plot for Sample Instance (High Risk Instance)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(target_shap_values[0], show=False)
        plt.title("SHAP Waterfall Explanation (Sample Customer)", fontsize=14)
        plt.tight_layout()
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()

        return target_shap_values.values, summary_path, waterfall_path
