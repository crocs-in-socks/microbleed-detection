import importlib.metadata
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import torch
from pydantic import BaseModel

from microbleednet.core import utils


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _source_state() -> tuple[Optional[str], Optional[bool]]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return revision, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _dependency_versions() -> dict[str, str]:
    names = ("microbleednet", "numpy", "nibabel", "pydantic", "scipy", "torch")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def capture_provenance(
    config: BaseModel | dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if isinstance(config, BaseModel):
        config_data = config.model_dump(mode="json")
    else:
        config_data = config

    device_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)

    source_revision, source_dirty = _source_state()
    return {
        "configuration": config_data,
        "seed": seed,
        "python_version": platform.python_version(),
        "dependencies": _dependency_versions(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def write_provenance(
    run_directory: Path,
    config: BaseModel | dict[str, Any],
    seed: int,
    device: torch.device,
) -> Path:
    path = run_directory / "provenance.json"
    utils.write_json_atomic(path, capture_provenance(config, seed, device))
    return path


def write_split_manifest(
    path: Path,
    splits: dict[str, Iterable[str]],
) -> Path:
    normalized = {
        name: sorted(set(subject_ids))
        for name, subject_ids in splits.items()
    }
    if set(normalized) != {"train", "validation", "tuning", "test"}:
        raise ValueError("splits must contain train, validation, tuning, and test")
    if any(not subject_id for values in normalized.values() for subject_id in values):
        raise ValueError("split subject IDs must be nonempty")

    manifest = {"splits": normalized}
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"split manifest already exists with different contents: {path}")
        return path

    utils.write_json_atomic(path, manifest)
    return path