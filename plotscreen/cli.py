"""Command line entry point."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import typer

from plotscreen import artifacts, conformal
from plotscreen.config import data_dir, load_settings

app = typer.Typer(add_completion=False, help="Screen farm plots for forest loss after a cut-off date.")
log = logging.getLogger("plotscreen")


def _setup(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("plotscreen").setLevel(logging.INFO)


@app.command()
def fetch(
    config: Path = typer.Option("aois.yaml", help="Areas and parameters."),
    baseline: bool = typer.Option(False, help="Also build the Sentinel-2 dNBR baseline."),
    verbose: bool = False,
) -> None:
    """Download embeddings and reference labels for every area in the config."""
    from plotscreen.pipeline import load_aoi

    _setup(verbose)
    settings, cache = load_settings(config), data_dir()
    for aoi in settings.aois:
        d = load_aoi(aoi, settings, cache)
        log.info("%s %s: %.1f%% of pixels lost after %d", aoi.role, aoi.name, 100 * d.labels.loss_after.mean(), settings.before_year)
        if baseline and aoi.role != "train":
            from plotscreen.sentinel2 import dnbr, periods

            before, after = periods(settings.before_year, settings.after_year)
            out = dnbr(aoi.name, aoi.bbox, before, after, d.before.transform, d.before.crs, d.before.shape, cache)
            log.info("  dNBR clear view on %.1f%% of pixels", 100 * np.isfinite(out).mean())


@app.command()
def run(
    config: Path = typer.Option("aois.yaml"),
    out: Path = typer.Option("results", help="Where metrics, model and plot table go."),
    docs: Path = typer.Option("docs", help="Where figures go."),
    seed: int = 0,
    repeats: int = 200,
    verbose: bool = False,
) -> None:
    """Fit the probe, score every plot, calibrate the triage and write the report."""
    from sklearn.metrics import roc_auc_score

    from plotscreen import pipeline, report, store

    _setup(verbose)
    settings, cache = load_settings(config), data_dir()
    res = pipeline.run(settings, cache, seed=seed, n_repeats=repeats)
    metrics, plots, vectors = res["metrics"], res["plots"], res["vectors"]

    table = store.to_table(
        {
            "plot_id": plots.plot_id.tolist(),
            "aoi": plots.aoi.tolist(),
            "split": plots.split.tolist(),
            "lon": plots.lon.tolist(),
            "lat": plots.lat.tolist(),
            "score": plots.score.astype(float).tolist(),
            "decision": plots.decision.astype(str).tolist(),
            "deforested": plots.deforested.tolist(),
        },
        vectors,
    )
    tbl = store.write_plots(cache / "lancedb", table)
    test = (plots.split == "test").to_numpy()
    votes = store.precedent_vote(tbl, vectors[test])
    truth = plots.deforested[test].to_numpy()
    in_review = (plots.decision[test] == conformal.REVIEW).to_numpy()
    metrics["precedents"] = {
        "k": 5,
        "n_queries": int(test.sum()),
        "vote_auc": float(roc_auc_score(truth, votes)),
        "majority_accuracy": float(((votes >= 0.5) == truth).mean()),
        "review_n": int(in_review.sum()),
        "review_majority_accuracy": float(((votes >= 0.5) == truth)[in_review].mean()) if in_review.any() else float("nan"),
    }

    artifacts.save_run(out, metrics, plots, vectors, res["model"])
    (out / "summary.md").write_text(report.summary_markdown(metrics))
    docs.mkdir(parents=True, exist_ok=True)
    report.results_figure(metrics, docs / "results.png")
    report.architecture_figure(docs / "architecture.png")
    showcase = _showcase(settings)
    d = pipeline.load_aoi(showcase, settings, cache)
    years = (settings.before_year, settings.after_year)
    report.map_figure(d.before.data, d.after.data, d.labels, plots, showcase.name, settings.plot_px, years, docs / "map.png")
    report.overview_figure(docs / "map.png", docs / "results.png", docs / "overview.png")
    typer.echo((out / "summary.md").read_text())


def _showcase(settings):
    areas = settings.by_role("eval")
    return areas[1 if len(areas) > 1 else 0]


@app.command(name="report")
def rebuild_report(
    config: Path = typer.Option("aois.yaml"),
    results: Path = typer.Option("results"),
    docs: Path = typer.Option("docs"),
) -> None:
    """Rebuild the summary and the figures from the results folder without rerunning anything."""
    import pandas as pd

    from plotscreen import report
    from plotscreen.embeddings import load_cube
    from plotscreen.reference import Labels

    settings, cache = load_settings(config), data_dir()
    metrics = json.loads((results / "metrics.json").read_text())
    (results / "summary.md").write_text(report.summary_markdown(metrics))
    docs.mkdir(parents=True, exist_ok=True)
    report.results_figure(metrics, docs / "results.png")
    report.architecture_figure(docs / "architecture.png")
    area = _showcase(settings)
    if (cache / f"{area.name}_labels.npz").exists():  # the map needs the cached cubes
        years = (settings.before_year, settings.after_year)
        before, after = (load_cube(cache, area.name, y).data for y in years)
        labels = Labels.load(cache / f"{area.name}_labels.npz")
        plots = pd.read_parquet(results / "plots.parquet")
        report.map_figure(before, after, labels, plots, area.name, settings.plot_px, years, docs / "map.png")
    if (docs / "map.png").exists():
        report.overview_figure(docs / "map.png", docs / "results.png", docs / "overview.png")
    typer.echo(f"summary and figures rebuilt from {results}")


@app.command()
def demo(
    results: Path = typer.Option("results", help="Folder with model.json, metrics.json and precedents.npz."),
    sample: Path = typer.Option("sample_data", help="Folder with chip.npz and pixels.npz."),
) -> None:
    """Offline walk through on the bundled sample: no download, a few seconds."""
    from sklearn.metrics import roc_auc_score

    from plotscreen import store
    from plotscreen.model import PixelModel, fit
    from plotscreen.plots import plot_scores, plot_truth
    from plotscreen.screen import load_thresholds

    metrics = json.loads((results / "metrics.json").read_text())
    s = metrics["settings"]
    model = PixelModel.load(results / "model.json")
    thr = load_thresholds(results / "metrics.json")
    chip = artifacts.load_chip(sample / "chip.npz")

    prob = model.predict_raster(chip["before"], chip["after"])
    scores = plot_scores(prob, s["plot_px"], s["top_k"])
    truth = plot_truth(chip["loss_after"], chip["forest_at_cutoff"], chip["valid"], s["plot_px"], s["min_loss_px"])
    decisions = conformal.decide(scores, thr)

    columns, vectors = artifacts.load_precedents(results / "precedents.npz")
    with tempfile.TemporaryDirectory() as tmp:
        tbl = store.write_plots(tmp, store.to_table(columns, vectors))
        signatures = store.change_signature(chip["before"], chip["after"], prob, s["plot_px"])
        votes = store.precedent_vote(tbl, signatures.reshape(-1, signatures.shape[-1])).reshape(scores.shape)

    typer.echo(f"Thresholds: clear below {thr.t_clear:.3f}, flag above {thr.t_flag:.3f} (alpha {thr.alpha}, beta {thr.beta})")
    typer.echo("row col  score  decision  reference_loss_px  similar_past_cases_deforested")
    for r in range(scores.shape[0]):
        for c in range(scores.shape[1]):
            typer.echo(f"{r:3d} {c:3d}  {scores[r, c]:.3f}  {decisions[r, c]:<8}  {int(truth.loss_px[r, c]):17d}  {votes[r, c]:.1f}")
    ev = conformal.evaluate(decisions.ravel(), truth.deforested.ravel())
    typer.echo(
        f"{ev['n']} plots, {ev['n_deforested']} deforested in the reference: "
        f"{ev['false_clear']} missed, {ev['false_flag']} wrongly flagged, {100 * ev['review_share']:.0f}% to review"
    )

    x, y = artifacts.load_pixels(sample / "pixels.npz")
    half = len(y) // 2
    small = fit(x[:half], y[:half])
    auc = roc_auc_score(y[half:], small.predict_proba(x[half:]))
    typer.echo(f"Probe refitted on {half} sample pixels: AUC {auc:.3f} on the other {len(y) - half}")


@app.command()
def sample(
    config: Path = typer.Option("aois.yaml"),
    out: Path = typer.Option("sample_data"),
    chip_px: int = 100,
    n_pixels: int = 4000,
    seed: int = 0,
) -> None:
    """Rebuild the bundled sample from the cache (maintainer task, needs `fetch` first)."""
    from plotscreen import pipeline
    from plotscreen.plots import plot_truth

    settings, cache = load_settings(config), data_dir()
    out.mkdir(parents=True, exist_ok=True)

    # A window of the area that neither the model nor the calibration has seen,
    # picked so that about 40% of its plots are deforested in the reference.
    area = (settings.by_role("transfer") or settings.by_role("eval"))[0]
    d = pipeline.load_aoi(area, settings, cache)
    truth = plot_truth(d.labels.loss_after, d.labels.forest_at_cutoff, d.labels.valid, settings.plot_px, settings.min_loss_px)
    side = chip_px // settings.plot_px
    best, best_gap = (0, 0), 1.0
    for r in range(0, truth.deforested.shape[0] - side + 1):
        for c in range(0, truth.deforested.shape[1] - side + 1):
            gap = abs(truth.deforested[r : r + side, c : c + side].mean() - 0.4)
            if gap < best_gap:
                best, best_gap = (r, c), gap
    r0, c0 = best[0] * settings.plot_px, best[1] * settings.plot_px
    rows, cols = slice(r0, r0 + chip_px), slice(c0, c0 + chip_px)
    artifacts.save_chip(out / "chip.npz", d.before.data, d.after.data, d.labels, d.before.transform, d.before.crs, rows, cols)

    per_area = n_pixels // max(len(settings.by_role("train")), 1)
    x, y = pipeline.training_sample(settings, cache, per_area, seed)
    order = np.random.default_rng(seed).permutation(len(y))
    artifacts.save_pixels(out / "pixels.npz", x[order], y[order])
    typer.echo(f"chip from {area.name} rows {r0}:{r0 + chip_px} cols {c0}:{c0 + chip_px}, {len(y)} pixels")


@app.command()
def check(
    plots_file: Path = typer.Argument(..., help="GeoJSON or GeoParquet with plot polygons."),
    results: Path = typer.Option("results"),
    out: Path = typer.Option("decisions.geojson"),
    verbose: bool = False,
) -> None:
    """Screen your own plot polygons. Needs network access to read the embeddings."""
    import geopandas as gpd

    from plotscreen import store
    from plotscreen.model import PixelModel
    from plotscreen.screen import load_thresholds, screen_polygons

    _setup(verbose)
    metrics = json.loads((results / "metrics.json").read_text())
    s = metrics["settings"]
    plots = gpd.read_parquet(plots_file) if plots_file.suffix == ".parquet" else gpd.read_file(plots_file)
    columns, vectors = artifacts.load_precedents(results / "precedents.npz")
    with tempfile.TemporaryDirectory() as tmp:
        tbl = store.write_plots(tmp, store.to_table(columns, vectors))
        decided = screen_polygons(
            plots,
            PixelModel.load(results / "model.json"),
            load_thresholds(results / "metrics.json"),
            s["before_year"],
            s["after_year"],
            s["top_k"],
            precedent_table=tbl,
        )
    decided.to_file(out, driver="GeoJSON")
    typer.echo(decided.drop(columns="geometry").to_string(index=False))
    typer.echo(f"written to {out}")


if __name__ == "__main__":
    app()
