"""Evidence store on LanceDB.

Every plot is stored with its decision and a 128-dimensional change signature
(the mean embedding difference of its pixels, weighted by the pixel
probability). The labelled calibration plots act as precedents: for any plot a
reviewer can pull the most similar past cases and see how they turned out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa

TABLE = "plots"
VECTOR_DIM = 128


def change_signature(before: np.ndarray, after: np.ndarray, weight: np.ndarray, plot_px: int) -> np.ndarray:
    """Weighted mean of (after - before) per plot, L2 normalised.

    Returns an array of shape (plot_rows, plot_cols, 128).
    """
    nr, nc = weight.shape[0] // plot_px, weight.shape[1] // plot_px
    out = np.zeros((nr, nc, before.shape[-1]), dtype=np.float32)
    for r in range(nr):
        rows = slice(r * plot_px, (r + 1) * plot_px)
        b = np.asarray(before[rows, : nc * plot_px], dtype=np.float32)
        a = np.asarray(after[rows, : nc * plot_px], dtype=np.float32)
        delta = np.nan_to_num(a - b).reshape(plot_px, nc, plot_px, -1)
        w = weight[rows, : nc * plot_px].reshape(plot_px, nc, plot_px, 1)
        out[r] = (delta * w).sum(axis=(0, 2)) / np.maximum(w.sum(axis=(0, 2)), 1e-6)
    norm = np.linalg.norm(out, axis=-1, keepdims=True)
    return out / np.maximum(norm, 1e-9)


def to_table(columns: dict[str, np.ndarray | list], vectors: np.ndarray) -> pa.Table:
    """Build the Arrow table LanceDB expects, with a fixed-size vector column."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIM:
        raise ValueError(f"vectors must be (n, {VECTOR_DIM})")
    arrays = {name: pa.array(values) for name, values in columns.items()}
    arrays["vector"] = pa.FixedSizeListArray.from_arrays(pa.array(vectors.ravel()), VECTOR_DIM)
    return pa.table(arrays)


def write_plots(db_path: str | Path, table: pa.Table):
    import lancedb

    db = lancedb.connect(str(db_path))
    return db.create_table(TABLE, data=table, mode="overwrite")


def open_plots(db_path: str | Path):
    import lancedb

    return lancedb.connect(str(db_path)).open_table(TABLE)


def precedents(table, vector: np.ndarray, k: int = 5, where: str = "split = 'calibration'") -> list[dict]:
    """The `k` most similar labelled plots, closest first."""
    query = table.search(np.asarray(vector, dtype=np.float32)).metric("cosine").limit(k)
    if where:
        query = query.where(where, prefilter=True)
    return query.to_list()


def precedent_vote(table, vectors: np.ndarray, k: int = 5) -> np.ndarray:
    """Share of deforested plots among the `k` nearest precedents of each vector."""
    votes = np.zeros(len(vectors), dtype=np.float32)
    for i, v in enumerate(vectors):
        hits = precedents(table, v, k)
        votes[i] = np.mean([h["deforested"] for h in hits]) if hits else np.nan
    return votes
