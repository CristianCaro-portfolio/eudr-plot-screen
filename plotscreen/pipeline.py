"""End-to-end experiment: fit the probe, score plots, calibrate the triage, measure it."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.metrics import average_precision_score, roc_auc_score

from plotscreen import conformal
from plotscreen.config import Aoi, Settings
from plotscreen.embeddings import check_same_grid, fetch_cube
from plotscreen.features import cosine_drift
from plotscreen.grid import block_ids, plot_bounds, split_blocks, to_plots
from plotscreen.model import PixelModel, fit, sample_pixels
from plotscreen.plots import plot_scores, plot_truth
from plotscreen.reference import Labels, fetch_labels
from plotscreen.store import change_signature

log = logging.getLogger(__name__)

METHODS = {"probe": "score", "cosine": "score_cosine", "dnbr": "score_dnbr"}


@dataclass
class AoiData:
    aoi: Aoi
    before: object
    after: object
    labels: Labels


def load_aoi(aoi: Aoi, settings: Settings, cache: Path, client=None) -> AoiData:
    """Embeddings of both years and reference labels, from cache or the network."""
    before = fetch_cube(aoi, settings.before_year, cache, client)
    after = fetch_cube(aoi, settings.after_year, cache, client)
    check_same_grid(before, after)
    labels = fetch_labels(
        aoi.name,
        cache,
        settings.hansen_version,
        before.transform,
        before.crs,
        before.shape,
        settings.cutoff_code,
        settings.after_year - 2000,
        settings.min_tree_cover,
    )
    return AoiData(aoi, before, after, labels)


def training_sample(settings: Settings, cache: Path, per_aoi: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for aoi in settings.by_role("train"):
        d = load_aoi(aoi, settings, cache)
        x, y = sample_pixels(d.before.data, d.after.data, d.labels.valid, d.labels.loss_after, per_aoi, rng)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def pixel_metrics(score: np.ndarray, labels: Labels) -> dict:
    ok = labels.valid & np.isfinite(score)
    y, s = labels.loss_after[ok], score[ok]
    pred = s >= 0.5
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    return {
        "auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "loss_share": float(y.mean()),
    }


def _cosine_raster(before: np.ndarray, after: np.ndarray, strip: int = 128) -> np.ndarray:
    out = np.zeros(before.shape[:2], dtype=np.float32)
    for r0 in range(0, before.shape[0], strip):
        out[r0 : r0 + strip] = np.nan_to_num(cosine_drift(before[r0 : r0 + strip], after[r0 : r0 + strip]))
    return out


def _dominant_loss_year(loss_year: np.ndarray, loss_after: np.ndarray, plot_px: int, codes: np.ndarray) -> np.ndarray:
    """Year in which each plot lost most pixels after the cut-off, 0 if none."""
    years = to_plots(np.where(loss_after, loss_year, 0), plot_px)
    counts = np.stack([(years == c).sum(axis=-1) for c in codes], axis=-1)
    best = codes[counts.argmax(axis=-1)] + 2000
    return np.where(counts.max(axis=-1) > 0, best, 0)


def score_aoi(
    d: AoiData,
    settings: Settings,
    model: PixelModel,
    cache: Path,
    with_extras: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, dict, np.ndarray]:
    """Plot table, change signatures, pixel metrics and the probability raster of one area."""
    px, k = settings.plot_px, settings.top_k
    prob = model.predict_raster(d.before.data, d.after.data)
    truth = plot_truth(d.labels.loss_after, d.labels.forest_at_cutoff, d.labels.valid, px, settings.min_loss_px)
    nr, nc = truth.deforested.shape
    rows, cols = np.indices((nr, nc))

    to_lonlat = Transformer.from_crs(d.before.crs, "EPSG:4326", always_xy=True)
    bounds = np.array([plot_bounds(d.before.transform, r, c, px) for r, c in zip(rows.ravel(), cols.ravel())])
    lon, lat = to_lonlat.transform((bounds[:, 0] + bounds[:, 2]) / 2, (bounds[:, 1] + bounds[:, 3]) / 2)

    table = pd.DataFrame(
        {
            "plot_id": [f"{d.aoi.name}:{r}:{c}" for r, c in zip(rows.ravel(), cols.ravel())],
            "aoi": d.aoi.name,
            "role": d.aoi.role,
            "row": rows.ravel(),
            "col": cols.ravel(),
            "block": block_ids(nr, nc).ravel(),
            "lon": np.round(lon, 6),
            "lat": np.round(lat, 6),
            "score": plot_scores(prob, px, k).ravel(),
            "deforested": truth.deforested.ravel(),
            "loss_px": truth.loss_px.ravel(),
            "forest_px": truth.forest_px.ravel(),
            "usable": truth.usable.ravel(),
        }
    )
    metrics = {"probe": pixel_metrics(prob, d.labels)}
    vectors = np.zeros((nr * nc, d.before.data.shape[-1]), dtype=np.float32)
    if with_extras:
        cos = _cosine_raster(d.before.data, d.after.data)
        table["score_cosine"] = plot_scores(cos, px, k).ravel()
        metrics["cosine"] = pixel_metrics(cos, d.labels)
        codes = np.arange(settings.cutoff_code + 1, settings.after_year - 2000 + 1)
        table["loss_year"] = _dominant_loss_year(d.labels.loss_year, d.labels.loss_after, px, codes).ravel()
        vectors = change_signature(d.before.data, d.after.data, prob, px).reshape(nr * nc, -1)
        dnbr_path = cache / f"{d.aoi.name}_dnbr.npz"
        if dnbr_path.exists():
            with np.load(dnbr_path) as z:
                dn = z["dnbr"]
            # A plot is only scored when its pixels were seen clearly in both periods.
            seen = plot_scores(np.isfinite(dn).astype(np.float32), px, px * px).ravel() >= 0.9
            scores = plot_scores(np.nan_to_num(dn, nan=-1.0), px, k).ravel()
            table["score_dnbr"] = np.where(seen, scores, np.nan)
            metrics["dnbr"] = pixel_metrics(dn, d.labels)
            metrics["dnbr"]["clear_view_share"] = float(np.isfinite(dn).mean())
    return table, vectors, metrics, prob


def assign_splits(table: pd.DataFrame, seed: int, test_share: float = 0.5) -> pd.Series:
    """Calibration or test for eval plots, by spatial block. Transfer plots keep their role."""
    split = pd.Series("transfer", index=table.index, dtype=object)
    rng = np.random.default_rng(seed)
    for _, part in table[table.role == "eval"].groupby("aoi"):
        in_test = split_blocks(part.block.to_numpy(), test_share, rng)
        split.loc[part.index] = np.where(in_test, "test", "calibration")
    return split


def triage(
    table: pd.DataFrame,
    column: str,
    alpha: float,
    beta: float,
    confidence: float | None = None,
    clustered: bool = False,
) -> tuple[conformal.Thresholds, dict]:
    """Calibrate on the calibration plots and evaluate on the test plots."""
    cal = table[(table.split == "calibration") & table[column].notna()]
    test = table[table.split == "test"]
    groups = (cal.aoi + ":" + cal.block.astype(str)).to_numpy() if clustered else None
    thr = conformal.calibrate(cal[column].to_numpy(), cal.deforested.to_numpy(), alpha, beta, confidence, groups)
    decisions = conformal.decide(test[column].to_numpy(dtype=float), thr)
    return thr, conformal.evaluate(decisions, test.deforested.to_numpy())


def repeated_splits(
    table: pd.DataFrame,
    column: str,
    alpha: float,
    beta: float,
    confidence: float | None,
    clustered: bool,
    n: int,
    seed: int,
) -> dict:
    """Redo the calibration/test split many times to see how the error rates spread."""
    base = table[table.role == "eval"].copy()
    rows = []
    for i in range(n):
        base["split"] = assign_splits(base, seed + 1000 + i)
        rows.append(triage(base, column, alpha, beta, confidence, clustered)[1])
    df = pd.DataFrame(rows)
    out: dict = {"n_splits": n, "confidence": confidence, "clustered": clustered}
    for col, target in (("false_clear_rate", alpha), ("false_flag_rate", beta), ("review_share", None)):
        out[col] = {
            "mean": float(df[col].mean()),
            "p05": float(df[col].quantile(0.05)),
            "p95": float(df[col].quantile(0.95)),
            "values": [round(float(v), 5) for v in df[col]],
        }
        if target is not None:
            out[col]["share_above_target"] = float((df[col] > target).mean())
    return out


RATES = ("false_clear_rate", "false_flag_rate", "review_share", "clear_share", "flag_share")


def repeated_grid(
    table: pd.DataFrame,
    methods: dict[str, str],
    alphas: tuple[float, ...],
    confidence: float | None,
    clustered: bool,
    n: int,
    seed: int,
) -> dict:
    """Every method at every tolerated error, averaged over the same random splits."""
    base = table[table.role == "eval"].copy()
    rows = []
    for i in range(n):
        base["split"] = assign_splits(base, seed + 1000 + i)
        for method, column in methods.items():
            for alpha in alphas:
                ev = triage(base, column, alpha, alpha, confidence, clustered)[1]
                rows.append({"method": method, "alpha": alpha, **{k: ev[k] for k in RATES}})
    df = pd.DataFrame(rows)
    out: dict = {}
    for (method, alpha), part in df.groupby(["method", "alpha"]):
        cell = {
            k: {"mean": float(part[k].mean()), "p05": float(part[k].quantile(0.05)), "p95": float(part[k].quantile(0.95))} for k in RATES
        }
        cell["missed_above_target"] = float((part.false_clear_rate > alpha).mean())
        out.setdefault(method, {})[f"{alpha:g}"] = cell
    return out


def aggregation_ablation(plots: pd.DataFrame, rasters: dict, settings: Settings) -> dict:
    """Same pixel probabilities, different ways of turning them into a plot score."""
    px = settings.plot_px
    variants = {"mean of all pixels": px * px, f"mean of top {settings.top_k}": settings.top_k, "single highest pixel": 1}
    out = {}
    for label, k in variants.items():
        table = plots[["plot_id", "aoi", "block", "split", "role", "deforested"]].copy()
        scores = {}
        for name, prob in rasters.items():
            grid = plot_scores(prob, px, k)
            rows, cols = np.indices(grid.shape)
            scores.update({f"{name}:{r}:{c}": v for r, c, v in zip(rows.ravel(), cols.ravel(), grid.ravel())})
        table["alt"] = table.plot_id.map(scores)
        test = table[table.split == "test"]
        _, ev = triage(table, "alt", settings.alpha, settings.beta, settings.confidence, settings.cluster_correction)
        out[label] = {"plot_auc": float(roc_auc_score(test.deforested, test.alt)), "review_share": ev["review_share"]}
    return out


def run(
    settings: Settings,
    cache: Path,
    seed: int = 0,
    train_per_aoi: int = 60_000,
    n_repeats: int = 200,
    label_sizes: tuple[int, ...] = (300, 3_000, 30_000, 180_000),
    label_seeds: tuple[int, ...] = (0, 1, 2),
    alphas: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10),
) -> dict:
    """Run the whole experiment on cached data and return everything worth keeping."""
    t0 = time.time()
    x, y = training_sample(settings, cache, train_per_aoi, seed)
    model = fit(x, y, seed=seed)
    log.info("probe fitted on %d pixels, %.1f%% loss", len(y), 100 * y.mean())

    targets = settings.by_role("eval") + settings.by_role("transfer")
    data = {a.name: load_aoi(a, settings, cache) for a in targets}
    tables, vectors, px_metrics, rasters = [], [], {}, {}
    for name, d in data.items():
        table, vec, metrics, prob = score_aoi(d, settings, model, cache)
        tables.append(table)
        vectors.append(vec)
        px_metrics[name] = metrics
        rasters[name] = prob
        log.info("%s scored: pixel AUC %.3f", name, metrics["probe"]["auc"])
    plots = pd.concat(tables, ignore_index=True)
    vectors = np.concatenate(vectors)
    keep = plots.usable.to_numpy()
    plots, vectors = plots[keep].reset_index(drop=True), vectors[keep]
    plots["split"] = assign_splits(plots, seed)

    methods = {m: c for m, c in METHODS.items() if c in plots.columns}
    conf, clustered = settings.confidence, settings.cluster_correction
    thr, _ = triage(plots, "score", settings.alpha, settings.beta, conf, clustered)
    plots["decision"] = conformal.decide(plots.score.to_numpy(), thr)

    result: dict = {
        "settings": settings.model_dump(mode="json"),
        "seed": seed,
        "n_training_pixels": int(len(y)),
        "training_loss_share": float(y.mean()),
        "thresholds": thr.to_dict(),
        "pixel": px_metrics,
        "plots": {
            split: {
                "n": int((plots.split == split).sum()),
                "deforested_share": float(plots.deforested[plots.split == split].mean()),
            }
            for split in ("calibration", "test", "transfer")
        },
        "plot_ranking": {},
        "triage": {},
        "transfer": {},
    }

    test = plots[plots.split == "test"]
    transfer = plots[plots.split == "transfer"]
    for method, column in methods.items():
        seen = test[column].notna()
        result["plot_ranking"][method] = {
            "auc": float(roc_auc_score(test.deforested[seen], test[column][seen])),
            "average_precision": float(average_precision_score(test.deforested[seen], test[column][seen])),
            "unscored_share": float(1 - seen.mean()),
        }
        result["triage"][method] = {}
        for alpha in alphas:
            m_thr, ev = triage(plots, column, alpha, alpha, conf, clustered)
            ev["t_clear"], ev["t_flag"] = m_thr.t_clear, m_thr.t_flag
            ev["false_clear_upper95"] = conformal.upper_bound(ev["false_clear"], ev["n_deforested"])
            result["triage"][method][f"{alpha:g}"] = ev
        m_thr, _ = triage(plots, column, settings.alpha, settings.beta, conf, clustered)
        if len(transfer):
            dec = conformal.decide(transfer[column].to_numpy(dtype=float), m_thr)
            result["transfer"][method] = conformal.evaluate(dec, transfer.deforested.to_numpy())

    # Same area, thresholds learned on half of its own blocks.
    if len(transfer):
        local = transfer.copy()
        local["role"] = "eval"
        local["split"] = assign_splits(local, seed)
        result["transfer"]["probe_recalibrated_locally"] = triage(local, "score", settings.alpha, settings.beta, conf, clustered)[1]

    modes = {"on_average": (None, False)}
    if conf is not None:
        modes["with_confidence"] = (conf, False)
        if clustered:
            modes["with_confidence_clustered"] = (conf, True)
    result["repeated_splits"] = {
        name: repeated_splits(plots, "score", settings.alpha, settings.beta, c, g, n_repeats, seed) for name, (c, g) in modes.items()
    }
    result["triage_repeated"] = repeated_grid(plots, methods, alphas, conf, clustered, n_repeats, seed)
    result["aggregation"] = aggregation_ablation(plots, rasters, settings)

    # What the errors look like.
    dec = plots.decision
    false_flags = plots[(plots.split == "test") & (dec == conformal.FLAG) & ~plots.deforested]
    result["error_analysis"] = {
        "false_flags": int(len(false_flags)),
        "false_flags_with_some_loss": int((false_flags.loss_px > 0).sum()),
        "false_flags_median_loss_px": float(false_flags.loss_px.median()) if len(false_flags) else 0.0,
        "by_loss_year": {},
    }
    positives = plots[(plots.split == "test") & plots.deforested]
    for year, part in positives.groupby("loss_year"):
        share = part.decision.value_counts(normalize=True)
        result["error_analysis"]["by_loss_year"][str(int(year))] = {
            "n": int(len(part)),
            **{k: float(share.get(k, 0.0)) for k in (conformal.CLEAR, conformal.REVIEW, conformal.FLAG)},
        }

    # How many labelled pixels the probe needs.
    result["label_efficiency"] = []
    eval_data = [data[a.name] for a in settings.by_role("eval")]
    for size in label_sizes:
        for s in label_seeds:
            if size >= len(y):
                if s != label_seeds[0]:
                    continue
                sub_model, n_used = model, len(y)
            else:
                idx = _stratified_subset(y, size, np.random.default_rng(100 + s))
                sub_model, n_used = fit(x[idx], y[idx], seed=s), size
            parts = [score_aoi(d, settings, sub_model, cache, with_extras=False)[0] for d in eval_data]
            sub = pd.concat(parts, ignore_index=True)
            sub = sub[sub.usable].reset_index(drop=True)
            sub["split"] = assign_splits(sub, seed)
            _, ev = triage(sub, "score", settings.alpha, settings.beta, conf, clustered)
            t = sub[sub.split == "test"]
            result["label_efficiency"].append(
                {
                    "labelled_pixels": int(n_used),
                    "seed": int(s),
                    "plot_auc": float(roc_auc_score(t.deforested, t.score)),
                    "review_share": ev["review_share"],
                    "false_clear_rate": ev["false_clear_rate"],
                    "false_flag_rate": ev["false_flag_rate"],
                }
            )
            log.info("label curve: %d pixels seed %d review %.3f", n_used, s, ev["review_share"])

    result["runtime_s"] = round(time.time() - t0, 1)
    return {"metrics": result, "plots": plots, "vectors": vectors, "model": model, "rasters": rasters}


def _stratified_subset(y: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Random subset that is guaranteed to contain both classes."""
    idx = rng.choice(len(y), size=size, replace=False)
    if len(np.unique(y[idx])) < 2:
        missing = 1 - int(y[idx][0])
        idx[0] = rng.choice(np.flatnonzero(y == missing))
    return idx
