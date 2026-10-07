"""Aggregate pixel rasters to plots."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plotscreen.grid import to_plots


def plot_scores(pixel_score: np.ndarray, plot_px: int, top_k: int) -> np.ndarray:
    """Mean of the `top_k` highest pixel scores inside each plot.

    A plot is deforested when at least `top_k` pixels were lost, so the score
    asks how confident the model is about the `top_k` most suspicious pixels and
    ignores the rest of the plot.
    """
    blocks = to_plots(pixel_score, plot_px)
    k = min(top_k, blocks.shape[-1])
    top = np.partition(blocks, -k, axis=-1)[..., -k:]
    return top.mean(axis=-1)


@dataclass
class PlotTruth:
    deforested: np.ndarray  # reference label per plot
    loss_px: np.ndarray  # pixels lost after the cut-off
    forest_px: np.ndarray  # forest pixels at the cut-off
    usable: np.ndarray  # enough valid pixels to judge the plot


def plot_truth(
    loss_after: np.ndarray,
    forest_at_cutoff: np.ndarray,
    valid: np.ndarray,
    plot_px: int,
    min_loss_px: int,
    min_valid_share: float = 0.9,
) -> PlotTruth:
    loss_px = to_plots(loss_after, plot_px).sum(axis=-1)
    forest_px = to_plots(forest_at_cutoff, plot_px).sum(axis=-1)
    valid_share = to_plots(valid, plot_px).mean(axis=-1)
    return PlotTruth(
        deforested=loss_px >= min_loss_px,
        loss_px=loss_px,
        forest_px=forest_px,
        usable=valid_share >= min_valid_share,
    )
