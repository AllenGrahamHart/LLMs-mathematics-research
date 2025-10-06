"""Configuration management for LLM Mathematics Research."""

import os
from pathlib import Path
import yaml
from typing import Dict, Any


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    # Look for config.yaml in the current working directory first
    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config

    # Fall back to package default location (project root)
    package_root = Path(__file__).parent.parent.parent
    default_config = package_root / "config.yaml"
    if default_config.exists():
        return default_config

    raise FileNotFoundError(
        "config.yaml not found. Please ensure config.yaml exists in your "
        "current working directory or the project root."
    )


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = get_config_path()
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Global config instance
CONFIG = load_config()
