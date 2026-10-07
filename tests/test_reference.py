import numpy as np
import pytest

from plotscreen.reference import Labels, build_labels, granule_for


@pytest.mark.parametrize(
    "lon, lat, expected",
    [(-72.4, 2.0, "10N_080W"), (-71.75, 3.25, "10N_080W"), (-60.5, -3.2, "00N_070W"), (10.2, 49.9, "50N_010E"), (-75.0, 10.0, "20N_080W")],
)
def test_granule_names(lon, lat, expected):
    assert granule_for(lon, lat) == expected


def test_labels_respect_the_cutoff():
    treecover = np.array([[90, 90, 90, 10, 90, 90]], dtype=np.uint8)
    lossyear = np.array([[0, 15, 22, 23, 26, 21]], dtype=np.uint8)
    datamask = np.array([[1, 1, 1, 1, 1, 2]], dtype=np.uint8)
    lab = build_labels(treecover, lossyear, datamask, cutoff_code=20, last_code=25, min_tree_cover=30)
    # stable forest, lost before the cut-off, lost after, never forest, lost too late to be seen, water
    assert lab.loss_after.tolist() == [[False, False, True, False, False, False]]
    assert lab.forest_at_cutoff.tolist() == [[True, False, True, False, True, False]]
    assert lab.valid.tolist() == [[True, True, True, True, True, False]]


def test_labels_roundtrip(tmp_path):
    lab = build_labels(np.full((4, 4), 80, np.uint8), np.eye(4, dtype=np.uint8) * 22, np.ones((4, 4), np.uint8), 20, 25)
    lab.save(tmp_path / "l.npz")
    again = Labels.load(tmp_path / "l.npz")
    assert (again.loss_after == lab.loss_after).all() and again.loss_year.dtype == np.uint8
