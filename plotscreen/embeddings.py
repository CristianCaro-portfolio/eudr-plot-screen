"""Read TESSERA embeddings for a bounding box and cache them on disk.

TESSERA publishes one 128-dimensional vector per 10 m pixel per year. The cubes
are read from the public Zarr store through geotessera, so only the pixels of
the requested box are transferred.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from affine import Affine

from plotscreen.config import Aoi

log = logging.getLogger(__name__)

EMBEDDING_DIM = 128


@dataclass
class Cube:
    """Embeddings of one area for one year on the native UTM grid."""

    data: np.ndarray  # (rows, cols, 128), float16
    transform: Affine
    crs: str
    dataset: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape[0], self.data.shape[1]


class GridMismatch(RuntimeError):
    """Two years of the same area did not land on the same pixel grid."""


def _paths(cache: Path, name: str, year: int) -> tuple[Path, Path]:
    return cache / f"{name}_{year}.npy", cache / f"{name}_{year}.json"


def load_cube(cache: Path, name: str, year: int) -> Cube:
    npy, meta_path = _paths(cache, name, year)
    meta = json.loads(meta_path.read_text())
    return Cube(
        data=np.load(npy, mmap_mode="r"),
        transform=Affine(*meta["transform"]),
        crs=meta["crs"],
        dataset=meta["dataset"],
    )


def fetch_cube(aoi: Aoi, year: int, cache: Path, client=None, retries: int = 3) -> Cube:
    """Return the cube for `aoi` and `year`, downloading it only if needed."""
    npy, meta_path = _paths(cache, aoi.name, year)
    if npy.exists() and meta_path.exists():
        return load_cube(cache, aoi.name, year)

    if client is None:
        from geotessera import GeoTesseraZarr

        client = GeoTesseraZarr()

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            mosaic, transform, crs = client.read_region(aoi.bbox, year=year)
            break
        except Exception as exc:  # network errors surface as many types here
            last_error = exc
            log.warning("read failed for %s %s (attempt %d): %s", aoi.name, year, attempt, exc)
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"could not read {aoi.name} {year}") from last_error

    if mosaic.ndim != 3 or mosaic.shape[2] != EMBEDDING_DIM:
        raise ValueError(f"unexpected embedding shape {mosaic.shape}")

    cache.mkdir(parents=True, exist_ok=True)
    np.save(npy, mosaic.astype(np.float16))
    dataset = getattr(client, "dataset", None)
    meta = {
        "aoi": aoi.name,
        "bbox": list(aoi.bbox),
        "year": year,
        "transform": list(transform)[:6],
        "crs": str(crs),
        "dataset": f"tessera-{getattr(dataset, 'version', '?')}-{getattr(dataset, 'variant', '?')}",
        "shape": list(mosaic.shape),
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return load_cube(cache, aoi.name, year)


def check_same_grid(a: Cube, b: Cube) -> None:
    if a.shape != b.shape or a.transform != b.transform or a.crs != b.crs:
        raise GridMismatch(f"{a.shape} {a.transform} {a.crs} vs {b.shape} {b.transform} {b.crs}")
