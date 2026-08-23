from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionSummary:
    """Paths and counts an inference run returns to its caller.

    Carries no arrays — just the locations of the artifacts written to disk and
    how many candidate components were found — so a frozen dataclass rather than
    a Pydantic model is the right fit for this in-memory return value.
    """

    subject_id: str
    mask_path: Path
    probability_path: Path
    components_path: Path
    component_count: int

    def __post_init__(self) -> None:
        if not self.subject_id.strip():
            raise ValueError("subject_id must be nonempty")
        if self.component_count < 0:
            raise ValueError("component_count must be non-negative")
