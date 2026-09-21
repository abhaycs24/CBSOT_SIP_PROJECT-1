import os
import joblib
from typing import Tuple, Any, Optional


class ModelSerializer:
    """Serializes and deserializes trained models and preprocessor pipelines using Joblib."""

    def __init__(self, artifacts_dir: str = "models/artifacts"):
        if not os.path.isabs(artifacts_dir):
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            artifacts_dir = os.path.join(base_dir, artifacts_dir)
        self.artifacts_dir = artifacts_dir
        self.model_path = os.path.join(artifacts_dir, "model.joblib")
        self.preprocessor_path = os.path.join(artifacts_dir, "preprocessor.joblib")

    def save(self, model: Any, preprocessor: Any) -> Tuple[str, str]:
        os.makedirs(self.artifacts_dir, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(preprocessor, self.preprocessor_path)
        return self.model_path, self.preprocessor_path

    def load(self) -> Tuple[Optional[Any], Optional[Any]]:
        model, preprocessor = None, None
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
        if os.path.exists(self.preprocessor_path):
            preprocessor = joblib.load(self.preprocessor_path)
        return model, preprocessor
