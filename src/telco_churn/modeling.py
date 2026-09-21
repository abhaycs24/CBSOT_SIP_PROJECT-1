import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, Any


class ChurnPredictor:
    """Manages model training, hyperparameter search, and prediction using RandomForest."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        param_dist: Dict[str, Any] = None,
        n_iter: int = 5,
        cv: int = 3,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.param_dist = param_dist or {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
        }
        self.n_iter = n_iter
        self.cv = cv
        self.best_model: RandomForestClassifier = None

    def train_test_split(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42, stratify: Any = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)

    def fit_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        rf = RandomForestClassifier(random_state=self.random_state)
        rf_tuned = RandomizedSearchCV(
            rf,
            param_distributions=self.param_dist,
            n_iter=self.n_iter,
            cv=self.cv,
            random_state=self.random_state,
        )
        rf_tuned.fit(X_train, y_train)
        self.best_model = rf_tuned.best_estimator_
        return self.best_model

    def train_and_tune(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = self.train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        self.best_model = self.fit_model(X_train, y_train)
        y_pred = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:, 1]

        return X_train, X_test, y_train, y_test, y_pred, y_prob
