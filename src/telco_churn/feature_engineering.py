import pandas as pd
import numpy as np
from typing import Tuple


class FeatureEngineer:
    """Transforms raw Telco data by generating domain interaction features and nominal One-Hot Encoding."""

    def __init__(self, create_ratios: bool = True, create_contract_risk: bool = True):
        self.create_ratios = create_ratios
        self.create_contract_risk = create_contract_risk

    def transform(self, df: pd.DataFrame, target_column: str = "Churn Label") -> pd.DataFrame:
        df_feat = df.copy()

        # 1. Tenure-to-Monthly Charge Ratio
        if self.create_ratios and "Tenure Months" in df_feat.columns and "Monthly Charges" in df_feat.columns:
            df_feat["Tenure_Monthly_Ratio"] = df_feat["Tenure Months"] / (df_feat["Monthly Charges"] + 1e-5)

        # 2. Total Charges Deviation
        if "Total Charges" in df_feat.columns:
            mean_total = df_feat["Total Charges"].mean()
            df_feat["Total_Charges_Dev"] = df_feat["Total Charges"] - mean_total

        # 3. Contract Risk Index mapping
        if self.create_contract_risk and "Contract" in df_feat.columns:
            contract_risk_map = {
                "Month-to-month": 3.0,
                "One year": 2.0,
                "Two year": 1.0,
            }
            df_feat["Contract_Risk_Index"] = df_feat["Contract"].map(contract_risk_map).fillna(2.0)

        return df_feat
