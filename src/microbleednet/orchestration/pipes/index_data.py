import glob
import re
from pathlib import Path
from typing import Optional

from natsort import natsorted

from .. import manifests
from ..configs import SUBJECT_ID_PLACEHOLDER, IndexDataConfig
from ..layouts import DatasetLayout
from ..manifests import (
    ManifestStatus,
    RawDatasetManifest,
    RawSource,
    RawSubject,
)


def execute(config: IndexDataConfig) -> None:
    config.dataset_dir.mkdir(parents=True, exist_ok=True)

    now = manifests.timestamp()
    source, subjects, unmatched_volumes, unmatched_masks = index_source(config, now)

    layout = DatasetLayout()
    manifest_path = config.dataset_dir / layout.raw_manifest
    existing = (
        manifests.read_manifest(manifest_path, RawDatasetManifest)
        if manifest_path.is_file()
        else None
    )

    raw_manifest = merge_source(
        existing,
        source=source,
        subjects=subjects,
        unmatched_volumes=unmatched_volumes,
        unmatched_masks=unmatched_masks,
        now=now,
    )
    manifests.write_manifest(manifest_path, raw_manifest)


def index_source(
    config: IndexDataConfig, now: str
) -> tuple[RawSource, list[RawSubject], list[str], list[str]]:
    """Index one input/label directory pair into a source and its subjects.

    Volume/mask matching happens within this single source; accumulation across
    sources is handled by :func:`merge_source`.
    """
    volume_paths = compute_paths(config.input_dir, config.volume_pattern)
    mask_paths = (
        compute_paths(config.label_dir, config.mask_pattern)
        if config.label_dir is not None and config.mask_pattern is not None
        else []
    )

    volume_subject_map = build_subject_map(
        config.input_dir, volume_paths, config.volume_pattern
    )
    mask_subject_map = (
        build_subject_map(config.label_dir, mask_paths, config.mask_pattern)
        if config.label_dir is not None and config.mask_pattern is not None
        else {}
    )

    volume_ids = set(volume_subject_map)
    mask_ids = set(mask_subject_map)
    unmatched_volumes = sorted(volume_ids - mask_ids)
    unmatched_masks = sorted(mask_ids - volume_ids)
    if config.require_masks and (unmatched_volumes or unmatched_masks):
        raise ValueError(
            f"unmatched subjects: volumes={unmatched_volumes}, masks={unmatched_masks}"
        )

    def namespaced(subject_id: str) -> str:
        return f"{config.source_id}_{subject_id}" if config.source_id else subject_id

    subjects = [
        RawSubject(
            subject_id=namespaced(subject_id),
            volume_path=str(volume_subject_map[subject_id].resolve()),
            mask_path=(
                str(mask_subject_map[subject_id].resolve())
                if mask_subject_map.get(subject_id)
                else None
            ),
        )
        for subject_id in natsorted(volume_subject_map)
    ]
    unmatched_volumes = [namespaced(subject_id) for subject_id in unmatched_volumes]
    unmatched_masks = [namespaced(subject_id) for subject_id in unmatched_masks]

    source = RawSource(
        input_dir=str(config.input_dir.resolve()),
        label_dir=str(config.label_dir.resolve()) if config.label_dir else None,
        volume_pattern=config.volume_pattern,
        mask_pattern=config.mask_pattern,
        source_id=config.source_id,
        added_on=now,
    )
    return source, subjects, unmatched_volumes, unmatched_masks


def merge_source(
    existing: RawDatasetManifest | None,
    *, # to force following arguments to be called using keywords
    source: RawSource,
    subjects: list[RawSubject],
    unmatched_volumes: list[str],
    unmatched_masks: list[str],
    now: str,
) -> RawDatasetManifest:
    """Append a freshly indexed source to ``existing`` (or build the first one).

    Subject IDs are globally unique across sources: a new subject that collides
    with one already indexed is a hard error, so accumulation never silently
    drops or overwrites prior data.
    """
    prior_subjects = existing.subjects if existing else []
    collisions = {subject.subject_id for subject in prior_subjects} & {
        subject.subject_id for subject in subjects
    }
    if collisions:
        raise ValueError(
            "subjects already indexed in this dataset: "
            f"{sorted(collisions)}; index them into a fresh dataset directory."
        )

    return RawDatasetManifest(
        status=ManifestStatus.COMPLETE,
        created_at=existing.created_at if existing else now,
        updated_at=now,
        sources=[*(existing.sources if existing else []), source],
        subjects=natsorted(
            [*prior_subjects, *subjects], key=lambda subject: subject.subject_id
        ),
        unmatched_volumes=sorted(
            {*(existing.unmatched_volumes if existing else []), *unmatched_volumes}
        ),
        unmatched_masks=sorted(
            {*(existing.unmatched_masks if existing else []), *unmatched_masks}
        ),
    )


def build_subject_map(
    root_dir: Path,
    paths: list[Path],
    pattern: str,
) -> dict[str, Path]:
    subject_map: dict[str, Path] = {}
    for path in paths:
        subject_id = extract_subject_id(root_dir, path, pattern)
        if subject_id is None or not subject_id.strip():
            raise ValueError(f"path does not match pattern or has an empty ID: {path}")
        if subject_id in subject_map:
            raise ValueError(f"duplicate subject ID '{subject_id}' in {root_dir}")
        subject_map[subject_id] = path
    return subject_map


def compute_paths(dir: Path, pattern: str) -> list[Path]:
    pattern_parts = pattern.split(SUBJECT_ID_PLACEHOLDER)
    glob_pattern = "*".join(glob.escape(part) for part in pattern_parts)
    return list(dir.rglob(glob_pattern))


def extract_subject_id(root_dir: Path, path: Path, pattern: str) -> Optional[str]:
    clean_path = path.relative_to(root_dir)

    pattern_parts = pattern.split(SUBJECT_ID_PLACEHOLDER)
    escaped_parts = [re.escape(part) for part in pattern_parts]
    regex_pattern = "^" + "(.*?)".join(escaped_parts) + "$"
    match = re.match(regex_pattern, str(clean_path))
    return match.group(1) if match else None
