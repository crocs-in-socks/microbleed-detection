from pathlib import Path

import pytest
import typer
from pydantic import BaseModel, Field

from microbleednet.cli.utils import config_fields, load_config, parse_config
from microbleednet.orchestration.configs import IndexDataConfig


def test_load_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_config(tmp_path / "missing.toml")


def test_load_config_rejects_non_toml_suffix(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=".toml"):
        load_config(path)


def test_load_config_rejects_malformed_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("not = valid = toml", encoding="utf-8")
    with pytest.raises(ValueError, match="could not read configuration"):
        load_config(path)


def test_parse_config_reports_validation_error_as_bad_parameter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text("dataset_dir = 'dataset'\n", encoding="utf-8")  # missing fields
    with pytest.raises(typer.BadParameter, match="invalid configuration"):
        parse_config(path, IndexDataConfig)


def test_config_fields_recurses_into_nested_models() -> None:
    class Inner(BaseModel):
        knob: int = Field(default=1, description="An inner knob.")

    class Outer(BaseModel):
        inner: Inner = Field(default_factory=Inner)

    fields = config_fields(Outer)
    assert fields == [("inner", "knob", "1", "An inner knob.")]
