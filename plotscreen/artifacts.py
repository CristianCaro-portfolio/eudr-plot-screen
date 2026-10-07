"""Read and write the files a run leaves behind, and the small bundled sample."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from plotscreen.grid import pixel_to_xy
from plotscreen.model import PixelModel

PLOT_COLUMNS = [
    "plot_id", "aoi", "split", "lon", "lat", "score", "score_cosine", "score_dnbr",
    "deforested", "loss_px", "forest_px", "loss_year", "decision",
]  # fmt: skip


def quantise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """int8 values plus one float scale per vector, the layout TESSERA itself ships."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    scale = np.maximum(np.abs(x).max(axis=-1, keepdims=True), 1e-6) / 127.0
    return np.round(x / scale).astype(np.int8), scale.astype(np.float32)


def dequantise(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale


def save_run(out: Path, metrics: dict, plots: pd.DataFrame, vectors: np.ndarray, model: PixelModel) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    model.save(out / "model.json")
    columns = [c for c in PLOT_COLUMNS if c in plots.columns]
    table = plots[columns].copy()
    for column in ("score", "score_cosine", "score_dnbr"):
        if column in table:
            table[column] = table[column].astype(np.float32)
    table.to_parquet(out / "plots.parquet", index=False, compression="zstd")
    cal = (plots.split == "calibration").to_numpy()
    q, scale = quantise(vectors[cal])
    np.savez_compressed(
        out / "precedents.npz",
        plot_id=plots.plot_id[cal].to_numpy(dtype=str),
        deforested=plots.deforested[cal].to_numpy(),
        vector=q,
        scale=scale,
    )


def load_precedents(path: Path) -> tuple[dict, np.ndarray]:
    with np.load(path) as z:
        columns = {
            "plot_id": z["plot_id"].tolist(),
            "split": ["calibration"] * len(z["plot_id"]),
            "deforested": z["deforested"].tolist(),
        }
        return columns, dequantise(z["vector"], z["scale"])


def save_chip(path: Path, before, after, labels, transform, crs, rows: slice, cols: slice) -> None:
    """A small real window: both years of embeddings and the reference masks."""
    qb, sb = quantise(before[rows, cols])
    qa, sa = quantise(after[rows, cols])
    x0, y0 = pixel_to_xy(transform, cols.start, rows.start)
    np.savez_compressed(
        path,
        before=qb, before_scale=sb, after=qa, after_scale=sa,
        loss_after=labels.loss_after[rows, cols],
        forest_at_cutoff=labels.forest_at_cutoff[rows, cols],
        valid=labels.valid[rows, cols],
        transform=np.array([transform.a, transform.b, x0, transform.d, transform.e, y0]),
        crs=np.array(crs),
    )  # fmt: skip


def load_chip(path: Path) -> dict:
    with np.load(path) as z:
        return {
            "before": dequantise(z["before"], z["before_scale"]),
            "after": dequantise(z["after"], z["after_scale"]),
            "loss_after": z["loss_after"],
            "forest_at_cutoff": z["forest_at_cutoff"],
            "valid": z["valid"],
            "transform": z["transform"].tolist(),
            "crs": str(z["crs"]),
        }


def save_pixels(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """Labelled pixel pairs. Only before and after are kept, the difference is rebuilt."""
    dim = x.shape[1] // 3
    qb, sb = quantise(x[:, :dim])
    qa, sa = quantise(x[:, dim : 2 * dim])
    np.savez_compressed(path, before=qb, before_scale=sb, after=qa, after_scale=sa, label=y.astype(np.int8))


def load_pixels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    from plotscreen.features import pair_features

    with np.load(path) as z:
        before = dequantise(z["before"], z["before_scale"])
        after = dequantise(z["after"], z["after_scale"])
        return pair_features(before, after), z["label"]
