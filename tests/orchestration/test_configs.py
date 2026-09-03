import pytest
from pydantic import ValidationError

from microbleednet.orchestration.configs import IndexDataConfig, PreprocessConfig
from tests.support import make_index_config


@pytest.mark.parametrize("source_id", ["siteA", "site-a", "site_a", "123"])
def test_source_id_accepts_filesystem_safe_values(tmp_path, source_id: str) -> None:
    config = make_index_config(tmp_path, source_id=source_id)

    assert config.source_id == source_id


@pytest.mark.parametrize("source_id", ["site/A", "site A", "site@A", ""])
def test_source_id_rejects_unsafe_values(tmp_path, source_id: str) -> None:
    with pytest.raises(ValidationError, match="source_id"):
        make_index_config(tmp_path, source_id=source_id)


def test_source_id_is_required(tmp_path) -> None:
    with pytest.raises(ValidationError, match="source_id"):
        IndexDataConfig.model_validate(
            {
                "dataset_dir": tmp_path / "dataset",
                "input_dir": tmp_path,
                "volume_pattern": "{subject_id}.nii.gz",
                "require_masks": False,
            }
        )


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"volume_pattern": "volume.nii.gz"}, "volume_pattern"),
        (
            {
                "label_dir": "labels",
                "mask_pattern": "mask.nii.gz",
            },
            "mask_pattern",
        ),
    ],
)
def test_patterns_require_subject_id_placeholder(
    tmp_path, overrides: dict[str, object], field: str
) -> None:
    with pytest.raises(ValidationError, match=field):
        make_index_config(tmp_path, **overrides)


def test_pattern_accepts_multiple_subject_id_placeholders(tmp_path) -> None:
    pattern = "{subject_id}/{subject_id}_volume.nii.gz"

    config = make_index_config(tmp_path, volume_pattern=pattern)

    assert config.volume_pattern == pattern


def test_mask_pattern_required_when_label_dir_given(tmp_path) -> None:
    with pytest.raises(ValidationError, match="mask_pattern is required"):
        make_index_config(tmp_path, label_dir=tmp_path)


def test_label_dir_required_when_masks_are_required(tmp_path) -> None:
    with pytest.raises(ValidationError, match="label_dir is required"):
        IndexDataConfig(
            dataset_dir=tmp_path / "dataset",
            input_dir=tmp_path,
            volume_pattern="{subject_id}.nii.gz",
            source_id="test-source",
        )


def test_index_config_rejects_missing_input_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="input_dir"):
        make_index_config(tmp_path, input_dir=tmp_path / "missing")


def test_index_config_rejects_missing_label_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="label_dir"):
        make_index_config(tmp_path, label_dir=tmp_path / "missing")


def test_preprocess_config_rejects_missing_dataset_dir(tmp_path) -> None:
    with pytest.raises(ValidationError, match="dataset_dir"):
        PreprocessConfig(dataset_dir=tmp_path / "missing")
