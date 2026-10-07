import geopandas as gpd
import numpy as np
from shapely.geometry import box

from plotscreen import conformal
from plotscreen.model import fit, sample_pixels
from plotscreen.screen import polygon_score, screen_polygons
from tests.conftest import fake_area


def test_polygon_score():
    prob = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)
    inside = np.array([[True, True], [True, False]])
    assert polygon_score(prob, inside, top_k=2) == (np.float32(0.85), 3)
    score, n = polygon_score(prob, np.zeros_like(inside), top_k=2)
    assert np.isnan(score) and n == 0


class FakeClient:
    """Serves one simulated area for the first polygon and fails for the second."""

    def __init__(self, before, after, transform, crs, broken_lon):
        self.cubes = {2020: before.astype(np.float32), 2025: after.astype(np.float32)}
        self.transform, self.crs, self.broken_lon = transform, crs, broken_lon

    def read_region(self, bbox, year):
        if bbox[0] > self.broken_lon:
            raise ConnectionError("no data")
        return self.cubes[year], self.transform, self.crs


def test_unreadable_plot_is_sent_to_review():
    from affine import Affine

    rng = np.random.default_rng(3)
    before, after, labels = fake_area(rng, size=60)
    x, y = sample_pixels(before, after, labels.valid, labels.loss_after, 2000, rng)
    model = fit(x, y)
    transform = Affine(10, 0, 500000, 0, -10, 200000)  # UTM 18N, near lon -75, lat 1.8
    area = gpd.GeoSeries([box(500000, 199400, 500600, 200000)], crs="EPSG:32618").to_crs("EPSG:4326").iloc[0]
    far = box(-70.0, 1.0, -69.99, 1.01)
    plots = gpd.GeoDataFrame({"name": ["ok", "broken"]}, geometry=[area, far], crs="EPSG:4326")
    thr = conformal.Thresholds(0.2, 0.8, 0.05, 0.05, 100, 100)

    out = screen_polygons(plots, model, thr, 2020, 2025, top_k=50, client=FakeClient(before, after, transform, "EPSG:32618", -71.0))
    assert out.decision.tolist()[1] == conformal.REVIEW and np.isnan(out.score[1])
    assert out.pixels[0] > 3000 and out.estimated_loss_ha[0] > 0
    expected = conformal.FLAG if labels.loss_after.sum() >= 50 else conformal.CLEAR
    assert out.decision[0] == expected
