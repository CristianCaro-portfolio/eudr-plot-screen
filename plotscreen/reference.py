"""Reference labels from Hansen Global Forest Change, warped to the embedding grid.

Only the window that covers the area is read from the public bucket. The three
layers used are tree cover in 2000, year of loss and the land/water mask.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

from plotscreen.grid import grid_bounds

log = logging.getLogger(__name__)

HANSEN_URL = "https://storage.googleapis.com/earthenginepartners-hansen/{version}/Hansen_{version}_{layer}_{granule}.tif"
LAYERS = ("treecover2000", "lossyear", "datamask")
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def granule_for(lon: float, lat: float) -> str:
    """Name of the 10 degree granule that holds a point, e.g. 10N_080W."""
    top = math.ceil(lat / 10) * 10
    if top == lat:
        top += 10
    left = math.floor(lon / 10) * 10
    ns = "N" if top >= 0 else "S"
    ew = "E" if left >= 0 else "W"
    return f"{abs(top):02d}{ns}_{abs(left):03d}{ew}"


def read_layer_on_grid(
    layer: str,
    version: str,
    transform: Affine,
    crs: str,
    shape: tuple[int, int],
) -> np.ndarray:
    """Read one Hansen layer and resample it (nearest) to the target grid."""
    west, south, east, north = grid_bounds(transform, shape)
    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", west, south, east, north)
    granule = granule_for(lon0, lat1)
    if granule_for(lon1, lat0) != granule:
        raise ValueError("area spans more than one Hansen granule, split it first")
    url = HANSEN_URL.format(version=version, layer=layer, granule=granule)

    pad = 0.002
    with rasterio.Env(**GDAL_OPTIONS), rasterio.open(url) as src:
        window = from_bounds(lon0 - pad, lat0 - pad, lon1 + pad, lat1 + pad, src.transform)
        window = window.round_offsets().round_lengths()
        block = src.read(1, window=window)
        out = np.zeros(shape, dtype=np.uint8)
        reproject(
            source=block,
            destination=out,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return out


@dataclass
class Labels:
    """Per-pixel reference on the embedding grid."""

    valid: np.ndarray  # land pixel with data
    forest_at_cutoff: np.ndarray  # forest at the end of the cut-off year
    loss_after: np.ndarray  # forest in 2000 that was lost after the cut-off
    loss_year: np.ndarray  # year minus 2000, 0 where no loss

    def save(self, path: Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> Labels:
        with np.load(path) as z:
            return cls(**{k: z[k] for k in z.files})


def build_labels(
    treecover: np.ndarray,
    lossyear: np.ndarray,
    datamask: np.ndarray,
    cutoff_code: int,
    last_code: int,
    min_tree_cover: int = 30,
) -> Labels:
    """Turn the three Hansen layers into the masks the pipeline needs.

    `cutoff_code` and `last_code` are years minus 2000. Loss after `last_code`
    is ignored because there is no embedding that could have seen it.
    """
    valid = datamask == 1
    forest_2000 = treecover >= min_tree_cover
    lost_before = (lossyear >= 1) & (lossyear <= cutoff_code)
    loss_after = forest_2000 & (lossyear > cutoff_code) & (lossyear <= last_code) & valid
    return Labels(
        valid=valid,
        forest_at_cutoff=forest_2000 & ~lost_before & valid,
        loss_after=loss_after,
        loss_year=np.where(valid, lossyear, 0).astype(np.uint8),
    )


def fetch_labels(
    name: str,
    cache: Path,
    version: str,
    transform: Affine,
    crs: str,
    shape: tuple[int, int],
    cutoff_code: int,
    last_code: int,
    min_tree_cover: int = 30,
) -> Labels:
    path = cache / f"{name}_labels.npz"
    if path.exists():
        return Labels.load(path)
    layers = {layer: read_layer_on_grid(layer, version, transform, crs, shape) for layer in LAYERS}
    labels = build_labels(
        layers["treecover2000"],
        layers["lossyear"],
        layers["datamask"],
        cutoff_code,
        last_code,
        min_tree_cover,
    )
    cache.mkdir(parents=True, exist_ok=True)
    labels.save(path)
    return labels
