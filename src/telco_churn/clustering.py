import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict


class CustomerSegmentation:
    """Handles unsupervised customer segmentation using K-Means clustering."""

    def __init__(self, n_clusters: int = 4, elbow_k_max: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.elbow_k_max = elbow_k_max
        self.random_state = random_state
        self.kmeans_model = None

    def plot_elbow_method(self, X_scaled: np.ndarray, output_dir: str = "artifacts") -> Tuple[List[float], str]:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "elbow_method.png")

        wcss = []
        k_range = range(1, self.elbow_k_max + 1)
        for i in k_range:
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=self.random_state, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)

        fig = plt.figure(figsize=(9, 6))
        plt.plot(k_range, wcss, marker='o', linestyle='--', color='purple')
        plt.title('Elbow Method for Optimal Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return wcss, save_path

    def fit_predict(self, df: pd.DataFrame, X_scaled: np.ndarray, output_dir: str = "artifacts") -> Tuple[pd.DataFrame, str]:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "customer_segmentation.png")

        df_segmented = df.copy()
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=10
        )
        df_segmented['Customer_Segment'] = self.kmeans_model.fit_predict(X_scaled)

        fig = plt.figure(figsize=(10, 7))
        if 'Tenure Months' in df_segmented.columns and 'Monthly Charges' in df_segmented.columns:
            sns.scatterplot(
                data=df_segmented,
                x='Tenure Months',
                y='Monthly Charges',
                hue='Customer_Segment',
                palette='viridis',
                legend='full',
            )
            plt.title('Customer Segmentation Dashboard (K-Means)')
        else:
            plt.title('Customer Segmentation Dashboard (Required columns not found)')

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return df_segmented, save_path
