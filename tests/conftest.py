"""Shared fixtures. Tests never touch the network: cubes and labels are simulated here."""

from __future__ import annotations

import json

import numpy as np
import pytest
from affine import Affine

from plotscreen.config import Aoi, Settings
from plotscreen.reference import Labels

DIM = 128
TRANSFORM = Affine(10, 0, 500000, 0, -10, 200000)
CRS = "EPSG:32618"
# Forest and pasture look the same in every simulated area, as they do in a shared embedding space.
PROTOTYPES = np.random.default_rng(42).normal(0, 1, (2, DIM)).astype(np.float32)


def fake_area(rng: np.random.Generator, size: int = 120):
    """Two years of embeddings where cleared pixels move along a fixed direction."""
    forest, pasture = PROTOTYPES
    was_forest = np.zeros((size, size), dtype=bool)
    was_forest[:, : int(size * 0.7)] = True
    lost = np.zeros((size, size), dtype=bool)
    for _ in range(6):
        r, c = rng.integers(0, size - 30), rng.integers(0, int(size * 0.7) - 30)
        lost[r : r + rng.integers(8, 30), c : c + rng.integers(8, 30)] = True
    lost &= was_forest

    def cube(is_forest):
        base = np.where(is_forest[..., None], forest, pasture)
        return (base + rng.normal(0, 0.6, (size, size, DIM))).astype(np.float16)

    before, after = cube(was_forest), cube(was_forest & ~lost)
    loss_year = np.where(lost, 21 + (np.arange(size)[:, None] // 30) % 5, 0).astype(np.uint8)
    labels = Labels(valid=np.ones((size, size), dtype=bool), forest_at_cutoff=was_forest, loss_after=lost, loss_year=loss_year)
    return before, after, labels


@pytest.fixture
def settings() -> Settings:
    return Settings(
        aois=[
            Aoi(name="t1", bbox=(-72.4, 2.0, -72.39, 2.01), role="train"),
            Aoi(name="e1", bbox=(-72.5, 1.8, -72.49, 1.81), role="eval"),
            Aoi(name="e2", bbox=(-72.6, 1.8, -72.59, 1.81), role="eval"),
            Aoi(name="x1", bbox=(-71.8, 3.2, -71.79, 3.21), role="transfer"),
        ],
        alpha=0.1,
        beta=0.1,
    )


@pytest.fixture
def cache(tmp_path, settings):
    """A cache folder that looks like the result of `plotscreen fetch`."""
    rng = np.random.default_rng(7)
    for aoi in settings.aois:
        before, after, labels = fake_area(rng)
        for year, cube in ((settings.before_year, before), (settings.after_year, after)):
            np.save(tmp_path / f"{aoi.name}_{year}.npy", cube)
            meta = {"transform": list(TRANSFORM)[:6], "crs": CRS, "dataset": "simulated"}
            (tmp_path / f"{aoi.name}_{year}.json").write_text(json.dumps(meta))
        labels.save(tmp_path / f"{aoi.name}_labels.npz")
    return tmp_path
