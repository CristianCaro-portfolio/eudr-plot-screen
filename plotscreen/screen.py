"""Screen arbitrary plot polygons with a fitted probe and calibrated thresholds."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.features import geometry_mask

from plotscreen import conformal
from plotscreen.model import PixelModel
from plotscreen.store import precedents

log = logging.getLogger(__name__)


def polygon_score(prob: np.ndarray, inside: np.ndarray, top_k: int) -> tuple[float, int]:
    """Top-k mean of the pixel probabilities that fall inside the polygon."""
    values = prob[inside]
    if values.size == 0:
        return float("nan"), 0
    k = min(top_k, values.size)
    return float(np.partition(values, -k)[-k:].mean()), int(values.size)


def screen_polygons(
    plots: gpd.GeoDataFrame,
    model: PixelModel,
    thresholds: conformal.Thresholds,
    before_year: int,
    after_year: int,
    top_k: int,
    client=None,
    precedent_table=None,
) -> gpd.GeoDataFrame:
    """Read the embeddings under each polygon, score it and decide.

    Polygons that cannot be read get no score and are sent to review.
    """
    if plots.crs is None:
        raise ValueError("plots need a coordinate reference system")
    if client is None:
        from geotessera import GeoTesseraZarr

        client = GeoTesseraZarr()

    plots = plots.to_crs("EPSG:4326").reset_index(drop=True)
    scores, pixels, lost_ha, votes = [], [], [], []
    for geom in plots.geometry:
        try:
            before, transform, crs = client.read_region(geom.bounds, year=before_year)
            after, transform_a, _ = client.read_region(geom.bounds, year=after_year)
            if before.shape != after.shape or transform != transform_a:
                raise ValueError("years are on different grids")
            prob = model.predict_raster(before, after)
            local = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(crs).iloc[0]
            inside = ~geometry_mask([local], out_shape=prob.shape, transform=transform, all_touched=True)
            score, n = polygon_score(prob, inside, top_k)
            area = float((prob[inside] >= 0.5).sum()) * abs(transform.a * transform.e) / 10_000
            vote = float("nan")
            if precedent_table is not None and n:
                delta = np.nan_to_num(after - before)[inside]
                weight = prob[inside][:, None]
                vec = (delta * weight).sum(axis=0) / max(float(weight.sum()), 1e-6)
                hits = precedents(precedent_table, vec / max(float(np.linalg.norm(vec)), 1e-9))
                vote = float(np.mean([h["deforested"] for h in hits])) if hits else float("nan")
        except Exception as exc:  # fail closed: an unreadable plot is never cleared
            log.warning("plot could not be scored: %s", exc)
            score, n, area, vote = float("nan"), 0, float("nan"), float("nan")
        scores.append(score)
        pixels.append(n)
        lost_ha.append(area)
        votes.append(vote)

    plots["score"] = scores
    plots["pixels"] = pixels
    plots["estimated_loss_ha"] = np.round(lost_ha, 2)
    plots["precedent_vote"] = votes
    plots["decision"] = conformal.decide(np.array(scores, dtype=float), thresholds)
    return plots


def load_thresholds(metrics_path: Path) -> conformal.Thresholds:
    return conformal.Thresholds(**json.loads(metrics_path.read_text())["thresholds"])
