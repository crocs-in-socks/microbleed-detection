import numpy as np
import pytest

from microbleednet.core.datamodels import PreprocessResult


def test_preprocess_result_accepts_valid_arrays() -> None:
    image = np.ones((2, 3, 4))
    mask = np.zeros(image.shape, dtype=int)
    affine = np.eye(4)

    result = PreprocessResult(image=image, mask=mask, affine=affine)

    assert result.image is image
    assert result.mask is mask
    assert result.affine is affine


@pytest.mark.parametrize(
    ("image", "mask", "message"),
    [
        (np.ones((2, 3)), None, "image must be a 3D array"),
        (
            np.array([[[np.inf]]]),
            None,
            "image must contain only finite values",
        ),
        (
            np.ones((2, 2, 2)),
            np.zeros((1, 2, 2), dtype=int),
            "mask must match image shape",
        ),
    ],
)
def test_preprocess_result_rejects_invalid_arrays(
    image: np.ndarray, mask: np.ndarray | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PreprocessResult(image=image, mask=mask, affine=np.eye(4))