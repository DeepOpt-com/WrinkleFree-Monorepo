"""Load data configs from YAML files.

This module provides functions to load data configurations from YAML files
in the CheaperTraining configs directory. This is the single source of truth
for data configurations - consumer repos (Fairy2, 1.58Quant) should NOT have
their own data YAML configs.

Usage:
    from cheapertraining.data import load_data_config, list_available_configs

    # Load the default mixed_pretrain config
    config = load_data_config("mixed_pretrain")

    # List available configs
    available = list_available_configs()  # ["mixed_pretrain", "fineweb", "downstream"]
"""

from pathlib import Path
from typing import Any

import yaml

# Config directory: WrinkleFree-CheaperTraining/configs/data/
# This file is at: src/cheapertraining/data/config_loader.py
# So we go up 4 levels to repo root, then into configs/data
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = _REPO_ROOT / "configs" / "data"


def load_data_config(name: str = "mixed_pretrain") -> dict[str, Any]:
    """Load a data config by name.

    Args:
        name: Name of the config file (without .yaml extension).
              Available configs can be listed with list_available_configs().

    Returns:
        dict: The parsed YAML config as a dictionary.

    Raises:
        ValueError: If the config name is not found.
        FileNotFoundError: If the configs directory doesn't exist.
    """
    if not CONFIG_DIR.exists():
        raise FileNotFoundError(
            f"CheaperTraining configs directory not found at {CONFIG_DIR}. "
            "Make sure CheaperTraining is installed from source (not wheel)."
        )

    config_path = CONFIG_DIR / f"{name}.yaml"
    if not config_path.exists():
        available = list_available_configs()
        raise ValueError(
            f"Data config '{name}' not found. Available: {available}"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def list_available_configs() -> list[str]:
    """List available data config names.

    Returns:
        list[str]: Names of available configs (without .yaml extension).
    """
    if not CONFIG_DIR.exists():
        return []
    return sorted([p.stem for p in CONFIG_DIR.glob("*.yaml")])


def get_config_path(name: str = "mixed_pretrain") -> Path:
    """Get the full path to a config file.

    Args:
        name: Name of the config file (without .yaml extension).

    Returns:
        Path: Full path to the config file.
    """
    return CONFIG_DIR / f"{name}.yaml"
