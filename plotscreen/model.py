"""Pixel model: a linear probe on the embeddings of the two years.

The model is a standardised logistic regression. It is stored as plain JSON so
it can be reviewed, diffed and loaded without unpickling anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from plotscreen.features import finite_mask, pair_features


@dataclass
class PixelModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_raster(self, before: np.ndarray, after: np.ndarray, strip: int = 128) -> np.ndarray:
        """Probability of loss for every pixel, computed in row strips."""
        rows, cols = before.shape[:2]
        out = np.zeros((rows, cols), dtype=np.float32)
        for r0 in range(0, rows, strip):
            b = np.asarray(before[r0 : r0 + strip], dtype=np.float32)
            a = np.asarray(after[r0 : r0 + strip], dtype=np.float32)
            ok = finite_mask(b, a)
            x = pair_features(np.nan_to_num(b), np.nan_to_num(a)).reshape(-1, 3 * b.shape[-1])
            p = self.predict_proba(x).reshape(ok.shape)
            out[r0 : r0 + strip] = np.where(ok, p, 0.0)
        return out

    def save(self, path: Path) -> None:
        payload = {
            "kind": "standardised-logistic-regression",
            "n_features": int(self.coef.size),
            "mean": self.mean.round(6).tolist(),
            "scale": self.scale.round(6).tolist(),
            "coef": self.coef.round(6).tolist(),
            "intercept": round(float(self.intercept), 6),
        }
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> PixelModel:
        p = json.loads(path.read_text())
        return cls(
            mean=np.array(p["mean"], dtype=np.float32),
            scale=np.array(p["scale"], dtype=np.float32),
            coef=np.array(p["coef"], dtype=np.float32),
            intercept=float(p["intercept"]),
        )


def fit(x: np.ndarray, y: np.ndarray, c: float = 0.1, seed: int = 0) -> PixelModel:
    """Fit the probe. Classes are reweighted because loss pixels are the minority."""
    if len(np.unique(y)) < 2:
        raise ValueError("training labels must contain both classes")
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=300, random_state=seed)
    clf.fit(scaler.transform(x), y)
    return PixelModel(
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
        coef=clf.coef_[0].astype(np.float32),
        intercept=float(clf.intercept_[0]),
    )


def sample_pixels(
    before: np.ndarray,
    after: np.ndarray,
    valid: np.ndarray,
    target: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw `n` labelled pixels at random from the valid ones."""
    idx = np.flatnonzero(valid.ravel())
    if n < len(idx):
        idx = np.sort(rng.choice(idx, size=n, replace=False))
    r, c = np.unravel_index(idx, valid.shape)
    b = np.asarray(before[r, c], dtype=np.float32)
    a = np.asarray(after[r, c], dtype=np.float32)
    ok = finite_mask(b, a)
    return pair_features(b[ok], a[ok]), target[r, c][ok].astype(np.int8)
