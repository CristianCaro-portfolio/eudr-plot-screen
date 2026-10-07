"""Square plots on the pixel grid and the spatial blocks used to split them."""

from __future__ import annotations

import numpy as np
from affine import Affine


def crop_to_plots(shape: tuple[int, int], plot_px: int) -> tuple[int, int]:
    """Number of whole plots that fit in each direction."""
    return shape[0] // plot_px, shape[1] // plot_px


def to_plots(raster: np.ndarray, plot_px: int) -> np.ndarray:
    """Reshape a (rows, cols) raster into (plot_rows, plot_cols, pixels_per_plot)."""
    nr, nc = crop_to_plots(raster.shape[:2], plot_px)
    cropped = raster[: nr * plot_px, : nc * plot_px]
    blocks = cropped.reshape(nr, plot_px, nc, plot_px).swapaxes(1, 2)
    return blocks.reshape(nr, nc, plot_px * plot_px)


def pixel_to_xy(transform: Affine, col: float, row: float) -> tuple[float, float]:
    """Map coordinates of a pixel corner (col and row may be fractional)."""
    return (
        transform.a * col + transform.b * row + transform.c,
        transform.d * col + transform.e * row + transform.f,
    )


def grid_bounds(transform: Affine, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Bounds of a north-up grid: (west, south, east, north)."""
    west, north = pixel_to_xy(transform, 0, 0)
    east, south = pixel_to_xy(transform, shape[1], shape[0])
    return west, south, east, north


def plot_bounds(transform: Affine, row: int, col: int, plot_px: int) -> tuple[float, float, float, float]:
    """Bounds of one plot in the CRS of the grid: (west, south, east, north)."""
    west, north = pixel_to_xy(transform, col * plot_px, row * plot_px)
    east, south = pixel_to_xy(transform, (col + 1) * plot_px, (row + 1) * plot_px)
    return west, south, east, north


def block_ids(nr: int, nc: int, block: int = 5) -> np.ndarray:
    """Id of the spatial block each plot belongs to.

    Neighbouring plots look alike, so calibration and test are split by blocks
    of `block` x `block` plots (1 km with the default plot size) instead of by
    single plots.
    """
    rows = np.arange(nr)[:, None] // block
    cols = np.arange(nc)[None, :] // block
    return rows * ((nc + block - 1) // block) + cols


def split_blocks(blocks: np.ndarray, test_share: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask, True where the plot falls in a test block."""
    ids = np.unique(blocks)
    chosen = rng.permutation(ids)[: int(round(len(ids) * test_share))]
    return np.isin(blocks, chosen)
