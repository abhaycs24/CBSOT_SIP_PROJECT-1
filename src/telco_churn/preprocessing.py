import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict


class DataPreprocessor:
    """Handles data sanitization, missing value imputation, categorical encoding, and feature scaling."""

    def __init__(self, target_column: str = "Churn Label", id_column: str = "customerID"):
        self.target_column = target_column
        self.id_column = id_column
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        if 'Total Charges' in df_clean.columns:
            df_clean['Total Charges'] = df_clean['Total Charges'].replace(' ', np.nan)
            df_clean['Total Charges'] = df_clean['Total Charges'].astype(float)
            df_clean['Total Charges'] = df_clean['Total Charges'].fillna(df_clean['Total Charges'].median())

        if self.id_column in df_clean.columns:
            df_clean.drop(columns=[self.id_column], inplace=True)

        return df_clean

    def encode_and_scale(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray]:
        df_encoded = df.copy()

        categorical_cols = df_encoded.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col != self.target_column:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.label_encoders[col] = le

        if self.target_column in df_encoded.columns:
            le_target = LabelEncoder()
            df_encoded[self.target_column] = le_target.fit_transform(df_encoded[self.target_column])
            self.label_encoders[self.target_column] = le_target

            X = df_encoded.drop(columns=[self.target_column])
            y = df_encoded[self.target_column]
            X_scaled = self.scaler.fit_transform(X)
            return df_encoded, X, y, X_scaled

        X_scaled = self.scaler.fit_transform(df_encoded)
        return df_encoded, df_encoded, pd.Series(), X_scaled
