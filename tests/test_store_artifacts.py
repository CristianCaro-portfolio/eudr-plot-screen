import numpy as np
import pytest

from plotscreen import artifacts, store
from tests.conftest import CRS, TRANSFORM, fake_area


def test_quantise_roundtrip_is_close():
    x = np.random.default_rng(0).normal(0, 1, (50, 128)).astype(np.float32)
    q, scale = artifacts.quantise(x)
    assert q.dtype == np.int8 and scale.shape == (50, 1)
    assert np.abs(artifacts.dequantise(q, scale) - x).max() < 0.03


def test_chip_and_pixels_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    before, after, labels = fake_area(rng, size=60)
    artifacts.save_chip(tmp_path / "chip.npz", before, after, labels, TRANSFORM, CRS, slice(20, 60), slice(0, 40))
    chip = artifacts.load_chip(tmp_path / "chip.npz")
    assert chip["before"].shape == (40, 40, 128) and chip["crs"] == CRS
    assert chip["transform"][5] == TRANSFORM.f - 20 * 10  # origin moved 20 rows down
    assert (chip["loss_after"] == labels.loss_after[20:60, 0:40]).all()

    x = rng.normal(0, 1, (30, 384)).astype(np.float32)
    x[:, 256:] = x[:, 128:256] - x[:, :128]
    artifacts.save_pixels(tmp_path / "px.npz", x, np.arange(30) % 2)
    x2, y2 = artifacts.load_pixels(tmp_path / "px.npz")
    assert x2.shape == x.shape and np.abs(x2 - x).max() < 0.06 and y2.sum() == 15


def test_change_signature_points_along_the_change():
    rng = np.random.default_rng(0)
    before, after, labels = fake_area(rng, size=60)
    sig = store.change_signature(before, after, labels.loss_after.astype(np.float32), plot_px=20)
    assert sig.shape == (3, 3, 128)
    changed = sig.reshape(-1, 128)[labels.loss_after.reshape(3, 20, 3, 20).any(axis=(1, 3)).ravel()]
    assert np.allclose(np.linalg.norm(changed, axis=1), 1, atol=1e-4)
    if len(changed) > 1:  # every clearing moves in the same direction
        assert (changed @ changed[0] > 0.8).all()


def test_precedents_come_from_calibration_only(tmp_path):
    rng = np.random.default_rng(0)
    vectors = rng.normal(0, 1, (40, 128)).astype(np.float32)
    columns = {
        "plot_id": [f"p{i}" for i in range(40)],
        "split": ["calibration"] * 20 + ["test"] * 20,
        "deforested": [i % 2 == 0 for i in range(40)],
    }
    table = store.write_plots(tmp_path / "db", store.to_table(columns, vectors))
    hits = store.precedents(table, vectors[3], k=5)
    assert hits[0]["plot_id"] == "p3" and len(hits) == 5
    assert all(h["split"] == "calibration" for h in hits)
    # a test plot is never its own precedent
    assert "p30" not in [h["plot_id"] for h in store.precedents(table, vectors[30], k=5)]
    votes = store.precedent_vote(store.open_plots(tmp_path / "db"), vectors[:4])
    assert votes.shape == (4,) and ((votes >= 0) & (votes <= 1)).all()


def test_vectors_must_have_128_dims():
    with pytest.raises(ValueError):
        store.to_table({"plot_id": ["a"]}, np.zeros((1, 64), dtype=np.float32))
