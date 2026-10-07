import numpy as np
import pytest

from plotscreen.config import Aoi
from plotscreen.embeddings import GridMismatch, check_same_grid, fetch_cube
from tests.conftest import CRS, TRANSFORM


class FakeClient:
    """Stands in for GeoTesseraZarr. Fails `failures` times before answering."""

    dataset = None

    def __init__(self, failures=0, shape=(40, 50, 128)):
        self.failures, self.shape, self.calls = failures, shape, 0

    def read_region(self, bbox, year):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("temporary")
        return np.full(self.shape, year / 1000, dtype=np.float32), TRANSFORM, CRS


AOI = Aoi(name="a", bbox=(-72.4, 2.0, -72.3, 2.1), role="train")


def test_fetch_caches_on_disk(tmp_path):
    client = FakeClient()
    cube = fetch_cube(AOI, 2020, tmp_path, client)
    assert cube.shape == (40, 50) and cube.data.dtype == np.float16 and cube.crs == CRS
    again = fetch_cube(AOI, 2020, tmp_path, client)
    assert client.calls == 1 and again.transform == TRANSFORM


def test_fetch_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr("plotscreen.embeddings.time.sleep", lambda s: None)
    client = FakeClient(failures=2)
    assert fetch_cube(AOI, 2020, tmp_path, client).shape == (40, 50)
    assert client.calls == 3


def test_fetch_gives_up(tmp_path, monkeypatch):
    monkeypatch.setattr("plotscreen.embeddings.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        fetch_cube(AOI, 2020, tmp_path, FakeClient(failures=5))
    assert not list(tmp_path.glob("*.npy"))


def test_unexpected_shape_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        fetch_cube(AOI, 2020, tmp_path, FakeClient(shape=(40, 50, 64)))


def test_years_must_share_a_grid(tmp_path):
    a = fetch_cube(AOI, 2020, tmp_path, FakeClient())
    b = fetch_cube(AOI, 2025, tmp_path, FakeClient(shape=(41, 50, 128)))
    with pytest.raises(GridMismatch):
        check_same_grid(a, b)
