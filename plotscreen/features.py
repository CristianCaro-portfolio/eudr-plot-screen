"""Features built from the embeddings of the two years."""

from __future__ import annotations

import numpy as np


def pair_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Concatenate before, after and their difference along the last axis."""
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    return np.concatenate([before, after, after - before], axis=-1)


def cosine_drift(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """1 - cosine similarity between the two years. Needs no labels."""
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    dot = np.einsum("...d,...d->...", before, after)
    norm = np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1)
    return 1.0 - dot / np.maximum(norm, 1e-9)


def finite_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Pixels where both years have an embedding."""
    return np.isfinite(before[..., 0]) & np.isfinite(after[..., 0])
