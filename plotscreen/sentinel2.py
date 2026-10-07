"""Classic baseline: difference of the Normalised Burn Ratio from Sentinel-2.

For each period a few of the least cloudy scenes are read from the public COG
archive through a STAC search, masked with the scene classification layer and
combined with a per-pixel median. NBR drops sharply when forest is cleared.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds

from plotscreen.grid import grid_bounds

log = logging.getLogger(__name__)

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
# Scene classification: vegetation, bare soil, water, unclassified.
CLEAR_CLASSES = (4, 5, 6, 7)
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


def nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    total = nir + swir
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, (nir - swir) / total, np.nan)


def _read_on_grid(href: str, transform: Affine, crs: str, shape: tuple[int, int], resampling) -> np.ndarray:
    west, south, east, north = grid_bounds(transform, shape)
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds(crs, src.crs, west, south, east, north)
        window = from_bounds(left - 60, bottom - 60, right + 60, top + 60, src.transform)
        window = window.round_offsets().round_lengths()
        block = src.read(1, window=window, boundless=True, fill_value=0)
        out = np.zeros(shape, dtype=np.float32)
        reproject(
            source=block.astype(np.float32),
            destination=out,
            src_transform=src.window_transform(window),
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=resampling,
        )
    return out


def _offset(item) -> float:
    """Scenes processed since 2022 carry a +1000 offset unless already removed."""
    if item.properties.get("earthsearch:boa_offset_applied", False):
        return 0.0
    baseline = str(item.properties.get("s2:processing_baseline", "0"))
    try:
        return 1000.0 if float(baseline) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def nbr_composite(
    bbox: tuple[float, float, float, float],
    period: str,
    transform: Affine,
    crs: str,
    shape: tuple[int, int],
    max_scenes: int = 6,
    max_cloud: float = 60.0,
) -> tuple[np.ndarray, list[str]]:
    """Median NBR of the clearest scenes in `period` ("YYYY-MM-DD/YYYY-MM-DD")."""
    from pystac_client import Client

    search = Client.open(STAC_URL).search(
        collections=[COLLECTION],
        bbox=list(bbox),
        datetime=period,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=300,
    )
    items = sorted(search.items(), key=lambda it: it.properties["eo:cloud_cover"])
    stack, used, seen = [], [], set()
    with rasterio.Env(**GDAL_OPTIONS):
        for item in items:
            key = item.id.rsplit("_", 2)[0]  # same tile and date reprocessed
            if key in seen:
                continue
            try:
                scl = _read_on_grid(item.assets["scl"].href, transform, crs, shape, Resampling.nearest)
                clear = np.isin(scl.astype(np.uint8), CLEAR_CLASSES)
                if clear.mean() < 0.05:
                    continue
                off = _offset(item)
                nir = _read_on_grid(item.assets["nir08"].href, transform, crs, shape, Resampling.bilinear)
                swir = _read_on_grid(item.assets["swir22"].href, transform, crs, shape, Resampling.bilinear)
            except Exception as exc:  # a broken scene should not stop the composite
                log.warning("skipping %s: %s", item.id, exc)
                continue
            valid = clear & (nir > 0) & (swir > 0)
            layer = nbr(np.maximum(nir - off, 1.0), np.maximum(swir - off, 1.0))
            stack.append(np.where(valid, layer, np.nan).astype(np.float32))
            used.append(item.id)
            seen.add(key)
            if len(stack) >= max_scenes:
                break
    if not stack:
        return np.full(shape, np.nan, dtype=np.float32), used
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN pixels
        composite = np.nanmedian(np.stack(stack), axis=0)
    return composite.astype(np.float32), used


def periods(before_year: int, after_year: int) -> tuple[str, str]:
    """Same months in both years so the season does not look like change."""
    return f"{before_year}-08-01/{before_year}-12-31", f"{after_year}-08-01/{after_year}-12-31"


def dnbr(
    name: str,
    bbox: tuple[float, float, float, float],
    before_period: str,
    after_period: str,
    transform: Affine,
    crs: str,
    shape: tuple[int, int],
    cache: Path,
) -> np.ndarray:
    """NBR before minus NBR after. NaN where either composite has no clear view."""
    path = cache / f"{name}_dnbr.npz"
    if path.exists():
        with np.load(path) as z:
            return z["dnbr"]
    before, used_b = nbr_composite(bbox, before_period, transform, crs, shape)
    after, used_a = nbr_composite(bbox, after_period, transform, crs, shape)
    out = (before - after).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, dnbr=out, before_scenes=np.array(used_b), after_scenes=np.array(used_a))
    return out
