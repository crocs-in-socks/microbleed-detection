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


def require_keys(config: dict[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"configuration is missing required keys: {', '.join(missing)}")


def path_value(config: dict[str, Any], key: str, must_exist: bool = True) -> Path:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"configuration key '{key}' must be a non-empty path")
    path = Path(value)
    if must_exist and not path.exists():
        raise ValueError(f"configured path does not exist: {path}")
    return path
