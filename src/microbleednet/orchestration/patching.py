"""Turn preprocessed subjects into materialized training patches.

This is the orchestration-side glue over ``core.dataloading.patchers``: core
knows how to extract and write patches given *arrays*, but not what a subject is.
A ``Patcher`` resolves a ``PreprocessedSubject`` to its on-disk volume/mask,
extracts patches, and materializes them, accumulating one ``PatchRecord`` per
patch across a set of subjects.

The two extraction strategies differ only in ``_extract``: non-overlapping tiles
the whole volume, while target-centered runs the detector to find candidate
locations and centers a patch on each. Everything else -- path resolution, mask
validation, materialization -- is shared in the base.
"""

from pathlib import Path

import numpy as np
import torch

from ..core import io as core_io
from ..core import utils as core_utils
from ..core.common.models import CandidateDetector
from ..core.dataloading import patchers as core_patchers
from ..core.datamodels import ExtractedPatch, ModelArchitecture, PatchRecord
from ..core.engines import processor as core_processor
from ..core.postprocessing.components import component_centers, describe_components
from .manifests import PreprocessedSubject


def load_array(path: str) -> np.ndarray:
    """Load a NIfTI volume at ``path`` and return it as a numpy array."""
    return core_io.nifti_to_numpy(core_io.load_volume(Path(path)))


class Patcher:
    """Materialize training patches for a set of subjects.

    Subclasses implement ``_extract`` (subject arrays -> patches); the base owns
    subject resolution, the maskless-subject guard, and materialization to disk.
    """

    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def collect(
        self,
        subjects: list[PreprocessedSubject],
        patch_dir: Path,
        augmentation_factor: int,
    ) -> list[PatchRecord]:
        records: list[PatchRecord] = []
        for subject in subjects:
            records.extend(
                self._patch_subject(subject, patch_dir, augmentation_factor)
            )
        return records

    def _patch_subject(
        self,
        subject: PreprocessedSubject,
        patch_dir: Path,
        augmentation_factor: int,
    ) -> list[PatchRecord]:
        if subject.mask_path is None:
            raise ValueError(f"training subject has no mask: {subject.subject_id}")

        volume = load_array(subject.volume_path)
        mask = load_array(subject.mask_path)

        extracted = self._extract(volume, mask)
        return core_patchers.materialize_patches(
            extracted, patch_dir, subject.subject_id, augmentation_factor
        )

    def _extract(self, volume: np.ndarray, mask: np.ndarray) -> list[ExtractedPatch]:
        raise NotImplementedError


class NonOverlappingPatcher(Patcher):
    """Tile the whole volume into non-overlapping patches (detector stage)."""

    def _extract(self, volume: np.ndarray, mask: np.ndarray) -> list[ExtractedPatch]:
        return core_patchers.nonoverlapping_patcher(volume, mask, self.patch_size)


class TargetCenteredPatcher(Patcher):
    """Center one patch on each detector candidate (teacher/student stages).

    Loads its detector on first extraction and reuses it for all subjects. The
    stage drops the patcher after materializing its patches, reclaiming detector
    GPU memory before model fitting begins.
    """

    def __init__(
        self,
        patch_size: int,
        *,
        architecture: ModelArchitecture,
        checkpoint_path: Path,
        device: torch.device,
        threshold: float,
    ):
        super().__init__(patch_size)
        self.architecture = architecture
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.threshold = threshold
        self.detector: CandidateDetector | None = None

    def _load_detector(self) -> CandidateDetector:
        if self.detector is None:
            detector = CandidateDetector(self.architecture)
            core_io.load_model_weights(detector, self.device, self.checkpoint_path)
            self.detector = detector
        return self.detector

    def _extract(self, volume: np.ndarray, mask: np.ndarray) -> list[ExtractedPatch]:
        logits = core_processor.infer(self._load_detector(), self.device, volume)
        output = core_utils.microbleed_probability(logits)[0]  # drop batch
        candidate_mask = (output > self.threshold).astype(np.uint8)
        _, candidates = describe_components(candidate_mask)

        return core_patchers.target_centered_patcher(
            volume, mask, component_centers(candidates), self.patch_size
        )
