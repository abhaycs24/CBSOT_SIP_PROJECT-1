import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class EDADashboard:
    """Renders and saves exploratory data analysis charts."""

    def __init__(self, style: str = "whitegrid", palette: str = "muted"):
        sns.set_theme(style=style, palette=palette)

    def plot_dashboard(self, df: pd.DataFrame, output_dir: str = "artifacts") -> str:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "eda_dashboard.png")

        fig = plt.figure(figsize=(18, 12))

        # 1. Churn Ratio
        plt.subplot(2, 2, 1)
        if 'Churn Label' in df.columns:
            df['Churn Label'].value_counts().plot.pie(
                autopct='%1.1f%%', startangle=90, colors=['#5dade2', '#ec7063'], explode=(0, 0.1)
            )
            plt.title('Overall Churn Ratio')
        else:
            plt.title('Overall Churn Ratio (Churn Label not found)')

        # 2. Correlation Heatmap
        plt.subplot(2, 2, 2)
        numeric_df = df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Feature Correlation Heatmap')

        # 3. Tenure vs Monthly Charges
        plt.subplot(2, 2, 3)
        if 'Tenure Months' in df.columns and 'Monthly Charges' in df.columns and 'Churn Label' in df.columns:
            sns.scatterplot(data=df, x='Tenure Months', y='Monthly Charges', hue='Churn Label', alpha=0.6)
            plt.title('Tenure vs Monthly Charges')
        else:
            plt.title('Tenure vs Monthly Charges (Required columns not found)')

        # 4. Contract Impact on Churn
        plt.subplot(2, 2, 4)
        if 'Contract' in df.columns and 'Churn Label' in df.columns:
            sns.countplot(data=df, x='Contract', hue='Churn Label', palette='Set2')
            plt.title('Contract Type Impact on Churn')
        else:
            plt.title('Contract Type Impact on Churn (Required columns not found)')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path
