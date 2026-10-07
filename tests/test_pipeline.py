import json

import numpy as np
from typer.testing import CliRunner

from plotscreen import artifacts, conformal, pipeline, report


def test_run_end_to_end_on_simulated_cache(settings, cache, tmp_path):
    res = pipeline.run(settings, cache, seed=0, train_per_aoi=3000, n_repeats=5, label_sizes=(200, 3000), label_seeds=(0,), alphas=(0.1,))
    m, plots = res["metrics"], res["plots"]

    assert set(plots.split.unique()) == {"calibration", "test", "transfer"}
    assert set(plots.decision.unique()) <= {conformal.CLEAR, conformal.REVIEW, conformal.FLAG}
    assert len(res["vectors"]) == len(plots)
    # blocks never straddle calibration and test
    for _, part in plots[plots.split != "transfer"].groupby("aoi"):
        assert part.groupby("block").split.nunique().max() == 1
    # the simulated signal is easy, so ranking is close to perfect
    assert m["plot_ranking"]["probe"]["auc"] > 0.95
    assert m["pixel"]["e1"]["probe"]["auc"] > 0.95
    assert m["repeated_splits"]["on_average"]["n_splits"] == 5
    assert set(m["aggregation"]) == {"mean of all pixels", "mean of top 50", "single highest pixel"}
    assert [row["labelled_pixels"] for row in m["label_efficiency"]] == [200, 3000]
    assert "probe_recalibrated_locally" in m["transfer"]

    out = tmp_path / "results"
    artifacts.save_run(out, m, plots, res["vectors"], res["model"])
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["thresholds"]["n_pos"] > 0
    columns, vectors = artifacts.load_precedents(out / "precedents.npz")
    assert len(columns["plot_id"]) == (plots.split == "calibration").sum() == len(vectors)

    text = report.summary_markdown(m)
    assert "Triage, mean over 5 random" in text and "probe" in m["triage_repeated"] and "nan" not in text.lower().replace("n/a", "")
    report.results_figure(m, tmp_path / "r.png")
    report.architecture_figure(tmp_path / "a.png")
    d = pipeline.load_aoi(settings.by_role("eval")[0], settings, cache)
    report.map_figure(d.before.data, d.after.data, d.labels, plots, "e1", settings.plot_px, (2020, 2025), tmp_path / "m.png")
    assert all((tmp_path / f).stat().st_size > 10_000 for f in ("r.png", "a.png", "m.png"))


def test_dominant_loss_year():
    loss_year = np.zeros((20, 40), dtype=np.uint8)
    loss_year[:3, :20] = 22
    loss_year[3:10, :20] = 24
    out = pipeline._dominant_loss_year(loss_year, loss_year > 0, 20, np.arange(21, 26))
    assert out.tolist() == [[2024, 0]]


def test_demo_runs_on_the_bundled_sample():
    """Smoke test of the offline path with the real sample shipped in the repo."""
    from plotscreen.cli import app

    result = CliRunner().invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    assert "Thresholds: clear below" in result.output
    assert "Probe refitted on" in result.output
