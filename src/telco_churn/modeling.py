import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

# Disable optuna verbosity for clean CLI logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ChurnPredictor:
    """Manages multi-model gradient boosting training, Optuna tuning, and model comparison."""

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        models_to_benchmark: List[str] = None,
        n_trials: int = 15,
        timeout: int = 60,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.models_to_benchmark = models_to_benchmark or ["rf", "xgboost", "lightgbm", "catboost"]
        self.n_trials = n_trials
        self.timeout = timeout
        self.benchmark_models: Dict[str, Any] = {}
        self.benchmark_results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None

    def _objective(self, trial: optuna.Trial, model_type: str, X_train: pd.DataFrame, y_train: pd.Series) -> float:
        if model_type == "rf":
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.random_state,
            )
        elif model_type == "xgboost":
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=self.random_state,
                eval_metric="logloss",
            )
        elif model_type == "lightgbm":
            n_estimators = trial.suggest_int("n_estimators", 50, 200, step=25)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 63)
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=self.random_state,
                verbosity=-1,
            )
        elif model_type == "catboost":
            iterations = trial.suggest_int("iterations", 50, 200, step=25)
            depth = trial.suggest_int("depth", 3, 8)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            model = CatBoostClassifier(
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                random_seed=self.random_state,
                verbose=0,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        scores = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc")
        return scores.mean()

    def train_and_benchmark(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, Any, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        best_overall_score = -1.0

        for model_type in self.models_to_benchmark:
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, model_type, X_train, y_train),
                n_trials=self.n_trials,
                timeout=self.timeout,
            )

            best_params = study.best_params

            if model_type == "rf":
                model = RandomForestClassifier(**best_params, random_state=self.random_state)
            elif model_type == "xgboost":
                model = xgb.XGBClassifier(**best_params, random_state=self.random_state, eval_metric="logloss")
            elif model_type == "lightgbm":
                model = lgb.LGBMClassifier(**best_params, random_state=self.random_state, verbosity=-1)
            elif model_type == "catboost":
                model = CatBoostClassifier(**best_params, random_seed=self.random_state, verbose=0)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            roc_auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            self.benchmark_models[model_type] = model
            self.benchmark_results[model_type] = {
                "ROC-AUC": roc_auc,
                "Accuracy": acc,
                "F1-Score": f1,
                "Precision": prec,
                "Recall": rec,
                "Best_Optuna_CV_AUC": study.best_value,
            }

            if roc_auc > best_overall_score:
                best_overall_score = roc_auc
                self.best_model_name = model_type
                self.best_model = model

        benchmark_df = pd.DataFrame(self.benchmark_results).T.sort_values(by="ROC-AUC", ascending=False)
        best_y_pred = self.best_model.predict(X_test)
        best_y_prob = self.best_model.predict_proba(X_test)[:, 1]

        return X_train, X_test, y_train, y_test, benchmark_df, self.best_model, best_y_pred, best_y_prob
