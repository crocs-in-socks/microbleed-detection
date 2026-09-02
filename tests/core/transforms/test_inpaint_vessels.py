from types import SimpleNamespace

import numpy as np
import pytest

from microbleednet.core.transforms import inpaint_vessels


def test_apply_dilates_detected_vessels_before_inpainting(monkeypatch) -> None:
    volume = np.zeros((3, 3, 3))
    vessel_mask = np.zeros_like(volume, dtype=bool)
    vessel_mask[1, 1, 1] = True
    observed = {}
    monkeypatch.setattr(
        inpaint_vessels, "get_volume_vessel_mask", lambda _: vessel_mask
    )

    def fake_inpaint(source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        observed["mask"] = mask
        return source + 1

    monkeypatch.setattr(inpaint_vessels, "inpaint_with_neighborhood_mean", fake_inpaint)

    result = inpaint_vessels.apply(volume)

    assert observed["mask"].sum() > vessel_mask.sum()
    np.testing.assert_array_equal(result, np.ones_like(volume))


def test_get_volume_vessel_mask_processes_every_slice() -> None:
    volume = np.zeros((2, 2, 2))

    mask = inpaint_vessels.get_volume_vessel_mask(volume)

    np.testing.assert_array_equal(mask, volume)


@pytest.mark.parametrize(
    "image_slice",
    [np.zeros((2, 2)), np.ones((2, 2))],
)
def test_get_slice_vessel_mask_returns_empty_for_unusable_slice(
    image_slice: np.ndarray,
) -> None:
    mask = inpaint_vessels.get_slice_vessel_mask(image_slice)

    np.testing.assert_array_equal(mask, np.zeros_like(image_slice))


@pytest.mark.parametrize(
    ("clusters", "linearity"),
    [
        (np.array([0, 0, 0, 1]), np.arange(4).reshape((2, 2))),
        (np.array([0, 1, 1, 1]), np.zeros((2, 2))),
    ],
)
def test_get_slice_vessel_mask_selects_smaller_cluster_and_filters_regions(
    clusters: np.ndarray, linearity: np.ndarray, monkeypatch
) -> None:
    class FakeClusterer:
        def fit(self, features: np.ndarray) -> "FakeClusterer":
            assert features.shape == (4, 2)
            self.labels_ = clusters
            return self

    properties = [
        SimpleNamespace(label=1, eccentricity=0.95, solidity=0.9),
        SimpleNamespace(label=2, eccentricity=0.5, solidity=0.9),
        SimpleNamespace(label=3, eccentricity=0.5, solidity=0.4),
    ]
    monkeypatch.setattr(
        inpaint_vessels, "frangi", lambda *args, **kwargs: np.ones((2, 2))
    )
    monkeypatch.setattr(
        inpaint_vessels, "get_linearity_measure", lambda _: linearity.copy()
    )
    monkeypatch.setattr(inpaint_vessels, "KMeans", lambda **kwargs: FakeClusterer())
    monkeypatch.setattr(
        inpaint_vessels, "label", lambda _: np.array([[1, 2], [3, 0]])
    )
    monkeypatch.setattr(inpaint_vessels, "regionprops", lambda _: properties)

    mask = inpaint_vessels.get_slice_vessel_mask(np.array([[1.0, 2.0], [3.0, 4.0]]))

    np.testing.assert_array_equal(mask, np.array([[True, False], [True, False]]))


def test_get_linearity_measure_returns_nonnegative_image() -> None:
    linearity = inpaint_vessels.get_linearity_measure(
        np.arange(25, dtype=float).reshape((5, 5))
    )

    assert linearity.shape == (5, 5)
    assert np.all(linearity >= 0)


def test_inpaint_with_neighborhood_mean_fills_masked_voxel() -> None:
    volume = np.arange(27, dtype=float).reshape((3, 3, 3))
    mask = np.zeros_like(volume, dtype=bool)
    mask[1, 1, 1] = True

    inpainted = inpaint_vessels.inpaint_with_neighborhood_mean(volume, mask)

    assert inpainted[1, 1, 1] == pytest.approx(13.0)
    np.testing.assert_array_equal(inpainted[~mask], volume[~mask])


def test_inpaint_with_neighborhood_mean_rejects_unresolvable_mask() -> None:
    with pytest.raises(ValueError, match="unresolved voxels"):
        inpaint_vessels.inpaint_with_neighborhood_mean(
            np.ones((2, 2, 2)), np.ones((2, 2, 2), dtype=bool)
        )