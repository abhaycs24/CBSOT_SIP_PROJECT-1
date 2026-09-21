import pandas as pd
import numpy as np
from typing import Optional


class DataLoader:
    """Class responsible for ingesting or synthesizing Telco subscriber datasets."""

    def __init__(self, raw_data_path: Optional[str] = None):
        self.raw_data_path = raw_data_path

    def load_data(
        self,
        num_samples: int = 100,
        random_state: int = 42,
        missing_charges_count: int = 5,
    ) -> pd.DataFrame:
        if self.raw_data_path and pd.io.common.file_exists(self.raw_data_path):
            return pd.read_csv(self.raw_data_path)
        
        return self.generate_synthetic_data(
            num_samples=num_samples,
            random_state=random_state,
            missing_charges_count=missing_charges_count,
        )

    @staticmethod
    def generate_synthetic_data(
        num_samples: int = 100,
        random_state: int = 42,
        missing_charges_count: int = 5,
    ) -> pd.DataFrame:
        np.random.seed(random_state)
        data = {
            'customerID': [f'C{i}' for i in range(1, num_samples + 1)],
            'Gender': np.random.choice(['Male', 'Female'], num_samples),
            'Tenure Months': np.random.randint(1, 73, num_samples),
            'Monthly Charges': np.random.uniform(20, 120, num_samples),
            'Total Charges': np.random.uniform(50, 5000, num_samples).astype(str),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_samples),
            'Churn Label': np.random.choice([0, 1], num_samples),
        }
        df = pd.DataFrame(data)

        # Introduce whitespace in Total Charges for preprocessing testing
        for _ in range(missing_charges_count):
            idx = np.random.randint(0, len(df))
            df.loc[idx, 'Total Charges'] = ' '

        return df
