"""Manifests the orchestration layer writes and reads, with their durability contract.

A manifest is a typed, versioned record a pipe stage writes to advertise
what it produced and whether it finished. Every manifest shares one envelope — a
schema version, a stable type discriminator, a lifecycle status, and
creation/update timestamps — wrapped around a payload typed for that stage.

Rules enforced here (see ``ARCHITECTURE.md``):

- Every manifest carries ``schema_version``, ``manifest_type``, ``status``,
  ``created_at`` and ``updated_at``.
- A consumer may only read a ``complete`` manifest; a ``failed`` manifest must
  carry a nonempty error string.
- Models forbid unknown fields, are immutable, and never type a field as
  ``Any``.

The models and their read/write helpers live together because the behavior is
intrinsic to the type and layer-agnostic: the helpers depend only on
:mod:`microbleednet.orchestration.atomic_io`, so manifest persistence does not
pull in the ML stack.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from ..core.datamodels import FrozenModel
from . import atomic_io

SCHEMA_VERSION = 1


class ManifestStatus(str, Enum):
    """Lifecycle state of the stage that produced a manifest."""

    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class Manifest(FrozenModel):
    """Envelope shared by every manifest."""

    schema_version: Literal[1] = Field(
        default=SCHEMA_VERSION,
        description="Manifest schema version; readers reject other versions.",
    )
    status: ManifestStatus = Field(
        description="Whether the producing stage is running, complete, or failed.",
    )
    created_at: str = Field(description="ISO-8601 UTC time the manifest was created.")
    updated_at: str = Field(description="ISO-8601 UTC time of the last write.")
    error: str | None = Field(
        default=None,
        description="Failure detail; required when status is failed, otherwise unset.",
    )

    @model_validator(mode="after")
    def check_status_error(self) -> "Manifest":
        if self.status is ManifestStatus.FAILED and not (self.error or "").strip():
            raise ValueError("a failed manifest must carry a nonempty error string")
        if self.status is not ManifestStatus.FAILED and self.error is not None:
            raise ValueError("only a failed manifest may carry an error string")
        return self


class RawSubject(FrozenModel):
    """One indexed subject: a volume and its optional lesion mask."""

    subject_id: str = Field(description="Unique subject identifier.")
    volume_path: str = Field(description="Absolute path to the raw volume.")
    mask_path: str | None = Field(
        default=None, description="Absolute path to the lesion mask, if indexed."
    )


class RawSource(FrozenModel):
    """A directory pair and filename patterns that contributed subjects."""

    input_dir: str = Field(description="Directory the volumes were indexed from.")
    label_dir: str | None = Field(
        default=None, description="Directory the masks were indexed from, if any."
    )
    volume_pattern: str = Field(description="Glob/regex pattern matching volumes.")
    mask_pattern: str | None = Field(
        default=None, description="Pattern matching masks, if masks were indexed."
    )
    source_id: str | None = Field(
        default=None,
        description="Namespace prepended to this source's subject IDs, if any.",
    )
    added_on: str = Field(description="ISO-8601 time this source was indexed.")


class RawDatasetManifest(Manifest):
    """Manifest ``index-data`` writes after building the raw dataset."""

    manifest_type: Literal["raw_dataset"] = "raw_dataset"
    sources: list[RawSource] = Field(description="Sources that contributed subjects.")
    subjects: list[RawSubject] = Field(description="Indexed subjects.")
    unmatched_volumes: list[str] = Field(
        default_factory=list, description="Subject IDs with a volume but no mask."
    )
    unmatched_masks: list[str] = Field(
        default_factory=list, description="Subject IDs with a mask but no volume."
    )

    @model_validator(mode="after")
    def unique_subjects(self) -> "RawDatasetManifest":
        reject_duplicate_ids(subject.subject_id for subject in self.subjects)
        return self


def reject_duplicate_ids(subject_ids) -> None:
    seen: set[str] = set()
    for subject_id in subject_ids:
        if subject_id in seen:
            raise ValueError(f"duplicate subject ID in manifest: {subject_id!r}")
        seen.add(subject_id)


def timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for manifest ``created_at``/``updated_at``."""
    return datetime.now(timezone.utc).isoformat()


def write_manifest(path: Path, manifest: Manifest) -> None:
    """Serialize ``manifest`` to ``path`` atomically as versioned JSON."""
    atomic_io.write_json_atomic(path, manifest.model_dump(mode="json"))


def read_manifest[ManifestType: Manifest](
    path: Path,
    manifest_class: type[ManifestType],
    *,
    require_complete: bool = True,
) -> ManifestType:
    """Load and validate a manifest of ``manifest_class`` from ``path``.

    Rejects an unversioned or otherwise malformed manifest with an actionable
    error, and (by default) refuses any manifest that is not ``complete``.
    """
    payload = atomic_io.read_json(path)
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError(
            f"{path} is not a versioned manifest; regenerate it with the current "
            "pipeline (it predates the schema_version contract)."
        )
    try:
        manifest = manifest_class.model_validate(payload)
    except ValidationError as error:
        raise ValueError(f"invalid manifest at {path}:\n{error}") from error
    if require_complete and manifest.status is not ManifestStatus.COMPLETE:
        raise ValueError(
            f"manifest at {path} has status {manifest.status.value!r}; "
            "a consumer may only read a complete manifest."
        )
    return manifest
