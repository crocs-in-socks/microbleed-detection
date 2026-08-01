import json
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"configuration file does not exist: {path}")
    try:
        if path.suffix.lower() == ".toml":
            with path.open("rb") as config_file:
                config = tomllib.load(config_file)
        else:
            with path.open(encoding="utf-8") as config_file:
                config = json.load(config_file)
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read configuration {path}: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("configuration root must be an object")
    return config
