import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import hdbscan
import umap
from typing import Tuple, List, Dict, Any


class CustomerSegmentation:
    """Multi-algorithm customer segmentation (K-Means, GMM, HDBSCAN) with UMAP & Silhouette Metrics."""

    def __init__(self, n_clusters: int = 4, elbow_k_max: int = 6, random_state: int = 42):
        self.n_clusters = n_clusters
        self.elbow_k_max = elbow_k_max
        self.random_state = random_state
        self.best_algorithm_name: str = "kmeans"
        self.best_model: Any = None
        self.benchmark_results: Dict[str, Dict[str, float]] = {}

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

    def benchmark_clustering(
        self, X_scaled: np.ndarray, algorithms: List[str] = None
    ) -> pd.DataFrame:
        algorithms = algorithms or ["kmeans", "gmm", "hdbscan"]

        for alg in algorithms:
            if alg == "kmeans":
                model = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=10)
                labels = model.fit_predict(X_scaled)
            elif alg == "gmm":
                model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
                labels = model.fit_predict(X_scaled)
            elif alg == "hdbscan":
                model = hdbscan.HDBSCAN(min_cluster_size=5)
                labels = model.fit_predict(X_scaled)
            else:
                continue

            unique_labels = set(labels) - {-1}
            if len(unique_labels) > 1:
                valid_idx = labels != -1
                X_valid = X_scaled[valid_idx] if np.any(labels == -1) else X_scaled
                l_valid = labels[valid_idx] if np.any(labels == -1) else labels

                sil = silhouette_score(X_valid, l_valid)
                ch = calinski_harabasz_score(X_valid, l_valid)
                db = davies_bouldin_score(X_valid, l_valid)
            else:
                sil, ch, db = -1.0, 0.0, 999.0

            self.benchmark_results[alg] = {
                "Silhouette_Score": sil,
                "Calinski_Harabasz": ch,
                "Davies_Bouldin": db,
                "Num_Clusters": len(unique_labels),
            }

        benchmark_df = pd.DataFrame(self.benchmark_results).T.sort_values(by="Silhouette_Score", ascending=False)
        self.best_algorithm_name = benchmark_df.index[0]
        return benchmark_df

    def plot_umap_projections(
        self, df: pd.DataFrame, X_scaled: np.ndarray, output_dir: str = "artifacts"
    ) -> Tuple[pd.DataFrame, str]:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "umap_clusters.png")

        # Fit UMAP projection
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=self.random_state)
        embedding = reducer.fit_transform(X_scaled)

        # Fit Best Cluster Algorithm
        if self.best_algorithm_name == "gmm":
            model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
            labels = model.fit_predict(X_scaled)
        elif self.best_algorithm_name == "hdbscan":
            model = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = model.fit_predict(X_scaled)
        else:
            model = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=10)
            labels = model.fit_predict(X_scaled)

        df_segmented = df.copy()
        df_segmented['Customer_Segment'] = labels
        df_segmented['UMAP1'] = embedding[:, 0]
        df_segmented['UMAP2'] = embedding[:, 1]

        fig = plt.figure(figsize=(10, 7))
        sns.scatterplot(
            data=df_segmented,
            x='UMAP1',
            y='UMAP2',
            hue='Customer_Segment',
            palette='viridis',
            style='Customer_Segment',
            s=80,
            alpha=0.8,
        )
        plt.title(f'UMAP 2D Customer Segmentation Projection ({self.best_algorithm_name.upper()})')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return df_segmented, save_path

    def fit_predict(self, df: pd.DataFrame, X_scaled: np.ndarray, output_dir: str = "artifacts") -> Tuple[pd.DataFrame, str]:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "customer_segmentation.png")

        df_segmented = df.copy()
        self.best_model = KMeans(
            n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state, n_init=10
        )
        df_segmented['Customer_Segment'] = self.best_model.fit_predict(X_scaled)

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
