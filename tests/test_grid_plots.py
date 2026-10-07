import numpy as np
from affine import Affine

from plotscreen.grid import block_ids, crop_to_plots, plot_bounds, split_blocks, to_plots
from plotscreen.plots import plot_scores, plot_truth


def test_to_plots_keeps_pixels_together():
    raster = np.arange(6 * 8).reshape(6, 8)
    plots = to_plots(raster, 2)
    assert plots.shape == (3, 4, 4)
    assert sorted(plots[1, 2]) == sorted(raster[2:4, 4:6].ravel())


def test_partial_plots_are_dropped():
    assert crop_to_plots((45, 61), 20) == (2, 3)
    assert to_plots(np.zeros((45, 61)), 20).shape == (2, 3, 400)


def test_plot_bounds_in_metres():
    transform = Affine(10, 0, 500000, 0, -10, 200000)
    west, south, east, north = plot_bounds(transform, row=1, col=2, plot_px=20)
    assert (west, east) == (500400, 500600)
    assert (north, south) == (199800, 199600)
    assert (east - west) * (north - south) == 40_000  # 4 ha


def test_blocks_do_not_leak_between_splits():
    blocks = block_ids(20, 20, block=5)
    assert len(np.unique(blocks)) == 16
    in_test = split_blocks(blocks, 0.5, np.random.default_rng(0))
    assert set(blocks[in_test]).isdisjoint(blocks[~in_test])
    assert 0.3 < in_test.mean() < 0.7


def test_plot_score_is_top_k_mean():
    prob = np.zeros((20, 40), dtype=np.float32)
    prob[:5, :10] = 0.9  # 50 pixels in the first plot
    prob[0, 20:30] = 0.9  # 10 pixels in the second
    scores = plot_scores(prob, plot_px=20, top_k=50)
    assert scores.shape == (1, 2)
    assert np.isclose(scores[0, 0], 0.9)
    assert np.isclose(scores[0, 1], 0.9 * 10 / 50)


def test_plot_truth_uses_half_hectare_rule():
    loss = np.zeros((20, 40), dtype=bool)
    loss[:5, :10] = True  # exactly 50 pixels
    loss[0, 20:30] = True  # 10 pixels, below the line
    valid = np.ones_like(loss)
    valid[:, 30:] = False  # second plot only half observed
    truth = plot_truth(loss, np.ones_like(loss), valid, plot_px=20, min_loss_px=50)
    assert truth.deforested.tolist() == [[True, False]]
    assert truth.loss_px.tolist() == [[50, 10]]
    assert truth.usable.tolist() == [[True, False]]
