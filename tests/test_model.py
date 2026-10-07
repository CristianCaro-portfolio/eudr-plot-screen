import json

import numpy as np
import pytest

from plotscreen.features import cosine_drift, pair_features
from plotscreen.model import PixelModel, fit, sample_pixels
from tests.conftest import fake_area


def test_pair_features_and_drift():
    a = np.ones((3, 4, 128), dtype=np.float16)
    assert pair_features(a, a).shape == (3, 4, 384)
    assert np.allclose(cosine_drift(a, a), 0, atol=1e-6)
    assert np.allclose(cosine_drift(a, -a), 2, atol=1e-6)
    assert np.isfinite(cosine_drift(np.zeros((2, 128)), np.zeros((2, 128)))).all()


def test_probe_learns_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    before, after, labels = fake_area(rng)
    x, y = sample_pixels(before, after, labels.valid, labels.loss_after, 4000, rng)
    assert x.shape == (4000, 384) and set(np.unique(y)) == {0, 1}
    model = fit(x, y)
    prob = model.predict_raster(before, after)
    assert prob[labels.loss_after].mean() > 0.9 and prob[~labels.loss_after].mean() < 0.1

    model.save(tmp_path / "m.json")
    payload = json.loads((tmp_path / "m.json").read_text())
    assert payload["n_features"] == 384
    again = PixelModel.load(tmp_path / "m.json")
    assert np.allclose(again.predict_raster(before, after), prob, atol=1e-3)


def test_missing_embeddings_score_zero():
    rng = np.random.default_rng(1)
    before, after, labels = fake_area(rng, size=60)
    x, y = sample_pixels(before, after, labels.valid, labels.loss_after, 2000, rng)
    model = fit(x, y)
    after = after.astype(np.float32)
    after[:5] = np.nan
    prob = model.predict_raster(before, after)
    assert np.isfinite(prob).all() and (prob[:5] == 0).all()


def test_fit_needs_both_classes():
    with pytest.raises(ValueError):
        fit(np.zeros((10, 384), dtype=np.float32), np.zeros(10, dtype=np.int8))
