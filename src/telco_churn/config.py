import os
import yaml
from typing import Any, Dict


class Config:
    """Utility class for loading and accessing pipeline configuration parameters."""

    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found at: {yaml_path}")
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    @property
    def raw_config(self) -> Dict[str, Any]:
        return self._config
