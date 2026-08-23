from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unwrap_model(model: nn.Module) -> nn.Module:
    return cast(nn.Module, model._orig_mod if hasattr(model, "_orig_mod") else model)


def microbleed_probability(logits: torch.Tensor) -> np.ndarray:
    """Return the microbleed channel of two-class logits as a NumPy array."""
    return F.softmax(logits, dim=1)[:, 1].cpu().numpy()


def initialize_teacher_from_detector(detector: nn.Module, teacher: nn.Module) -> None:
    detector_state = unwrap_model(detector).state_dict()
    teacher_state = unwrap_model(teacher).state_dict()
    transferable = {
        key: value
        for key, value in detector_state.items()
        if key.startswith(("feature_extractor.", "segmentor."))
    }
    missing = [key for key in transferable if key not in teacher_state]
    if missing:
        raise RuntimeError(f"detector-to-teacher keys missing in teacher: {missing}")
    teacher_state.update(transferable)
    unwrap_model(teacher).load_state_dict(teacher_state, strict=True)
