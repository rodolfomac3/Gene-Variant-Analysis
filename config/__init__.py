from pathlib import Path
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Load config on import
CONFIG = load_config()