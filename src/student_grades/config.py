"""Configuration loading: no hard-coded paths elsewhere in the package."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Read the YAML configuration file.

    Args:
        path: Location of the YAML file (defaults to ``config.yaml`` at the root).

    Returns:
        The configuration as a dictionary.
    """
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)
