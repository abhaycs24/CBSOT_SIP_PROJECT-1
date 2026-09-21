import pandas as pd
import numpy as np
from typing import Tuple, List
from imblearn.over_sampling import SMOTENC, SMOTE


class ClassBalancer:
    """Handles dataset class rebalancing using SMOTE / SMOTE-NC for mixed continuous and nominal feature spaces."""

    def __init__(self, random_state: int = 42, k_neighbors: int = 3):
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def balance_training_data(
        self, X_train: pd.DataFrame, y_train: pd.Series, categorical_indices: List[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        if y_train.value_counts().min() <= self.k_neighbors:
            # Not enough minority samples for k_neighbors
            return X_train, y_train

        try:
            if categorical_indices and len(categorical_indices) > 0:
                smote = SMOTENC(
                    categorical_features=categorical_indices,
                    random_state=self.random_state,
                    k_neighbors=min(self.k_neighbors, y_train.value_counts().min() - 1),
                )
            else:
                smote = SMOTE(
                    random_state=self.random_state,
                    k_neighbors=min(self.k_neighbors, y_train.value_counts().min() - 1),
                )

            X_res, y_res = smote.fit_resample(X_train, y_train)
            res_df = pd.DataFrame(X_res, columns=X_train.columns)
            return res_df, pd.Series(y_res)
        except Exception as e:
            print(f"SMOTE-NC note (falling back to un-resampled split): {e}")
            return X_train, y_train
