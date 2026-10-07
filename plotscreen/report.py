"""Figures and the markdown summary of a run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

SURFACE, INK, MUTED, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b1"
SERIES = {"probe": "#2a78d6", "cosine": "#eb6834", "dnbr": "#1baf7a"}
NAMES = {"probe": "Embedding probe", "cosine": "Embedding drift, no labels", "dnbr": "Sentinel-2 dNBR"}
STATUS = {"clear": "#0ca30c", "review": "#fab219", "flag": "#d03b3b"}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True,
    }
)  # fmt: skip


def results_figure(metrics: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    alpha = metrics["settings"]["alpha"]

    ax = axes[0]
    grid = metrics["triage_repeated"]
    methods = [m for m in SERIES if m in grid]
    levels = sorted(grid["probe"], key=float, reverse=True)
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        cells = [grid[method][lv]["review_share"] for lv in levels]
        values = [100 * c["mean"] for c in cells]
        spread = [[100 * (c["mean"] - c["p05"]) for c in cells], [100 * (c["p95"] - c["mean"]) for c in cells]]
        x = np.arange(len(levels)) + (i - (len(methods) - 1) / 2) * width
        ax.bar(x, values, width * 0.9, color=SERIES[method], label=NAMES[method])
        ax.errorbar(x, values, yerr=spread, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        for xi, v, up in zip(x, values, spread[1]):
            ax.text(xi, v + up + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(np.arange(len(levels)), [f"{100 * float(lv):g}%" for lv in levels])
    ax.set_ylim(0, 118)
    ax.set_xlabel("tolerated error rate (missed, and wrongly flagged)")
    ax.set_ylabel("plots sent to manual review (%)")
    ax.set_title("Manual review needed for the same guarantee")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    curve = pd.DataFrame(metrics["label_efficiency"]).groupby("labelled_pixels").review_share
    mean, lo, hi = curve.mean() * 100, curve.min() * 100, curve.max() * 100
    ax.fill_between(mean.index, lo, hi, color=SERIES["probe"], alpha=0.15, linewidth=0)
    ax.plot(mean.index, mean, color=SERIES["probe"], linewidth=2, marker="o", markersize=6)
    for n, v in mean.items():
        ax.annotate(f"{v:.0f}%", (n, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_ylim(0, max(hi.max() * 1.25, 10))
    ax.set_xlabel("labelled pixels used to fit the probe")
    ax.set_ylabel(f"plots sent to review at {100 * alpha:g}% tolerated error (%)")
    ax.set_title("How many labels the probe needs")

    ax = axes[2]
    rep = metrics["repeated_splits"]
    modes = [("on_average", "on average", NEUTRAL)]
    final = [key for key in ("with_confidence_clustered", "with_confidence") if key in rep]
    if final:
        modes.append((final[0], f"{100 * rep[final[0]]['confidence']:g}% confidence", SERIES["probe"]))
    every = np.concatenate([100 * np.array(rep[key]["false_clear_rate"]["values"]) for key, _, _ in modes])
    bins = np.linspace(every.min(), every.max(), 30)
    for key, label, color in modes:
        values = 100 * np.array(rep[key]["false_clear_rate"]["values"])
        above = 100 * rep[key]["false_clear_rate"]["share_above_target"]
        ax.hist(
            values, bins=bins, color=color, edgecolor=SURFACE, linewidth=1, alpha=0.9, label=f"{label}: {above:.0f}% of splits above target"
        )
    top = ax.get_ylim()[1]
    ax.plot([100 * alpha] * 2, [0, top * 1.08], color=INK, linewidth=1.2, linestyle="--")
    ax.text(100 * alpha, top * 1.08, f" target {100 * alpha:g}%", va="top", fontsize=8)
    ax.set_ylim(0, top * 1.4)
    ax.set_xlabel("deforested test plots that were cleared (%)")
    ax.set_ylabel(f"splits (of {rep['on_average']['n_splits']})")
    ax.set_title("Missed deforestation across random splits")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _pca_rgb(cube: np.ndarray, basis: tuple | None = None, step: int = 2):
    """First three principal components as an RGB picture."""
    x = np.nan_to_num(np.asarray(cube[::step, ::step], dtype=np.float32))
    flat = x.reshape(-1, x.shape[-1])
    if basis is None:
        mean = flat.mean(axis=0)
        _, _, vt = np.linalg.svd(flat[:: max(len(flat) // 20000, 1)] - mean, full_matrices=False)
        proj = (flat - mean) @ vt[:3].T
        basis = (mean, vt[:3], np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0))
    mean, comps, lo, hi = basis
    proj = (flat - mean) @ comps.T
    rgb = np.clip((proj - lo) / (hi - lo), 0, 1)
    return rgb.reshape(x.shape[0], x.shape[1], 3), basis


def map_figure(before, after, labels, plots: pd.DataFrame, aoi: str, plot_px: int, years: tuple[int, int], path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    rgb_b, basis = _pca_rgb(before)
    rgb_a, _ = _pca_rgb(after, basis)
    axes[0].imshow(rgb_b)
    axes[0].set_title(f"Embeddings {years[0]} (3 of 128 dims)")
    axes[1].imshow(rgb_a)
    axes[1].set_title(f"Embeddings {years[1]}")

    ref = np.zeros(labels.loss_after.shape, dtype=np.uint8)
    ref[labels.forest_at_cutoff] = 1
    ref[labels.loss_after] = 2
    axes[2].imshow(ref[::2, ::2], cmap=ListedColormap(["#efeee9", "#9bbf9b", INK]), vmin=0, vmax=2, interpolation="nearest")
    axes[2].set_title(f"Reference: forest loss after {years[0]}")
    axes[2].legend(
        handles=[Patch(color="#efeee9", label="not forest"), Patch(color="#9bbf9b", label="forest"), Patch(color=INK, label="lost")],
        loc="lower left", fontsize=7, frameon=True, framealpha=0.9, ncols=3,
    )  # fmt: skip

    part = plots[plots.aoi == aoi].copy()
    if "row" not in part:  # the saved plot table keeps row and column inside the id
        part[["row", "col"]] = part.plot_id.str.split(":", expand=True)[[1, 2]].astype(int).to_numpy()
    grid = np.full((part.row.max() + 1, part.col.max() + 1), np.nan)
    order = ["clear", "review", "flag"]
    grid[part.row, part.col] = part.decision.map({k: i for i, k in enumerate(order)})
    axes[3].imshow(grid, cmap=ListedColormap([STATUS[k] for k in order]), vmin=0, vmax=2, interpolation="nearest")
    axes[3].set_title("Triage per 4 ha plot")
    counts = part.decision.value_counts()
    axes[3].legend(
        handles=[Patch(color=STATUS[k], label=f"{k} ({counts.get(k, 0)})") for k in order],
        loc="lower left", fontsize=7, frameon=True, framealpha=0.9, ncols=3,
    )  # fmt: skip
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for side in ax.spines.values():
            side.set_visible(False)
    place = ", ".join(word.capitalize() for word in reversed(aoi.split("_")))
    fig.suptitle(f"{place}: 11 km x 11 km of the Colombian Amazon frontier", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def architecture_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    boxes = [
        (0.2, 2.4, "TESSERA embeddings", "128 dims per 10 m pixel\n2020 and 2025, Zarr on S3", "#dbe9fb"),
        (0.2, 0.5, "Hansen GFC v1.13", "tree cover, loss year\nwindowed reads", "#efeee9"),
        (3.0, 1.45, "Pixel probe", "logistic regression on\nbefore, after, difference", "#dbe9fb"),
        (5.8, 1.45, "Plot score", "mean of the 50 most\nsuspicious pixels (0.5 ha)", "#dbe9fb"),
        (8.6, 1.45, "Conformal triage", "two thresholds from\ncalibration plots", "#dbe9fb"),
        (11.4, 2.75, "clear", "missed loss <= alpha", "#cfeccf"),
        (11.4, 1.45, "review", "a person decides", "#fdeab8"),
        (11.4, 0.15, "flag", "false alarms <= beta", "#f5cfcf"),
        (8.6, 3.2, "LanceDB", "precedents by similarity", "#efeee9"),
    ]
    for x, y, title, body, color in boxes:
        h = 0.85 if title in ("clear", "review", "flag", "LanceDB") else 1.25
        ax.add_patch(
            FancyBboxPatch((x, y), 2.3, h, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=color, edgecolor=MUTED, linewidth=0.8)
        )
        ax.text(x + 1.15, y + h - 0.28, title, ha="center", fontweight="bold", fontsize=10)
        ax.text(x + 1.15, y + h - 0.5, body, ha="center", va="top", fontsize=8, color=MUTED)
    arrow = dict(arrowstyle="->", color=MUTED, linewidth=1.2)
    for start, end in [
        ((2.5, 3.0), (3.0, 2.3)), ((2.5, 1.1), (3.0, 1.8)), ((5.3, 2.07), (5.8, 2.07)), ((8.1, 2.07), (8.6, 2.07)),
        ((10.9, 2.3), (11.4, 3.1)), ((10.9, 2.07), (11.4, 1.9)), ((10.9, 1.8), (11.4, 0.7)), ((9.75, 2.7), (9.75, 3.2)),
    ]:  # fmt: skip
        ax.annotate("", xy=end, xytext=start, arrowprops=arrow)
    ax.text(1.35, 1.95, "labels only for\nfitting and calibration", ha="center", fontsize=7.5, color=MUTED, style="italic")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def overview_figure(top: Path, bottom: Path, path: Path) -> None:
    """Stack the map and the results into one picture of the same width."""
    from PIL import Image

    images = [Image.open(top).convert("RGB"), Image.open(bottom).convert("RGB")]
    width = min(im.width for im in images)
    images = [im.resize((width, round(im.height * width / im.width))) for im in images]
    canvas = Image.new("RGB", (width, sum(im.height for im in images)), SURFACE)
    offset = 0
    for im in images:
        canvas.paste(im, (0, offset))
        offset += im.height
    canvas.save(path, optimize=True)


def _pct(v: float, digits: int = 1) -> str:
    return "n/a" if v != v else f"{100 * v:.{digits}f}%"


def summary_markdown(m: dict) -> str:
    s, alpha = m["settings"], m["settings"]["alpha"]
    lines = [
        "# Results",
        "",
        f"Embeddings {s['before_year']} and {s['after_year']}, reference {s['hansen_version']}, seed {m['seed']}.",
        f"Probe fitted on {m['n_training_pixels']:,} pixels from the train areas ({_pct(m['training_loss_share'])} loss).",
        "",
        "## Plots",
        "",
        "| split | plots | deforested |",
        "|---|---:|---:|",
    ]
    for split, v in m["plots"].items():
        lines.append(f"| {split} | {v['n']:,} | {_pct(v['deforested_share'])} |")
    lines += ["", "## Pixel level, per area", "", "| area | method | AUC | average precision | IoU at 0.5 |", "|---|---|---:|---:|---:|"]
    for area, by_method in m["pixel"].items():
        for method, v in by_method.items():
            iou = f"{v['iou']:.3f}" if method == "probe" else ""
            lines.append(f"| {area} | {method} | {v['auc']:.3f} | {v['average_precision']:.3f} | {iou} |")
    lines += [
        "",
        "## Plot ranking on the test plots",
        "",
        "| method | AUC | average precision | plots without a score |",
        "|---|---:|---:|---:|",
    ]
    for method, v in m["plot_ranking"].items():
        lines.append(f"| {method} | {v['auc']:.3f} | {v['average_precision']:.3f} | {_pct(v['unscored_share'])} |")

    def spread(cell: dict, digits: int = 1) -> str:
        return f"{_pct(cell['mean'], digits)} ({_pct(cell['p05'], digits)} to {_pct(cell['p95'], digits)})"

    n_splits = next(iter(m["repeated_splits"].values()))["n_splits"]
    lines += [
        "",
        f"## Triage, mean over {n_splits} random calibration/test splits (5th to 95th percentile)",
        "",
        "| method | tolerated error | cleared | flagged | review | missed | wrongly flagged | splits with missed above target |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in [k for k in SERIES if k in m["triage_repeated"]]:
        for level, v in sorted(m["triage_repeated"][method].items(), key=lambda kv: -float(kv[0])):
            lines.append(
                f"| {method} | {_pct(float(level), 0)} | {_pct(v['clear_share']['mean'])} | {_pct(v['flag_share']['mean'])} "
                f"| {spread(v['review_share'])} | {spread(v['false_clear_rate'])} | {spread(v['false_flag_rate'])} "
                f"| {_pct(v['missed_above_target'])} |"
            )
    lines += [
        "",
        f"## Triage on the single split behind `plots.parquet` (seed {m['seed']})",
        "",
        "| method | tolerated error | cleared | flagged | review | missed (cleared but deforested) | wrongly flagged |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, by_level in m["triage"].items():
        for level, v in sorted(by_level.items(), key=lambda kv: -float(kv[0])):
            lines.append(
                f"| {method} | {_pct(float(level), 0)} | {_pct(v['clear_share'])} | {_pct(v['flag_share'])} | {_pct(v['review_share'])} "
                f"| {_pct(v['false_clear_rate'])} ({v['false_clear']} of {v['n_deforested']}) | {_pct(v['false_flag_rate'])} |"
            )
    conf = s.get("confidence")
    mode = f"with {_pct(conf, 0)} confidence" if conf else "on average"
    if conf and s.get("cluster_correction"):
        mode += " and corrected for clustered plots"
    lines.insert(3, f"Tolerated error {_pct(alpha, 0)} missed and {_pct(s['beta'], 0)} wrongly flagged, calibrated {mode}.")
    for rep in m["repeated_splits"].values():
        title = "on average" if rep["confidence"] is None else f"with {_pct(rep['confidence'], 0)} confidence"
        if rep["confidence"] is not None:
            title += ", corrected for clustered plots" if rep["clustered"] else ", plots taken as independent"
        lines += [
            "",
            f"## Probe calibrated {title}: {rep['n_splits']} splits at {_pct(alpha, 0)}",
            "",
            "| rate | mean | 5th percentile | 95th percentile | splits above target |",
            "|---|---:|---:|---:|---:|",
        ]
        for col, label in (("false_clear_rate", "missed"), ("false_flag_rate", "wrongly flagged"), ("review_share", "review")):
            above = _pct(rep[col]["share_above_target"]) if "share_above_target" in rep[col] else ""
            lines.append(f"| {label} | {_pct(rep[col]['mean'], 2)} | {_pct(rep[col]['p05'], 2)} | {_pct(rep[col]['p95'], 2)} | {above} |")
    lines += ["", "## Plot score: how to aggregate pixels", "", "| aggregation | plot AUC | review share |", "|---|---:|---:|"]
    for label, v in m["aggregation"].items():
        lines.append(f"| {label} | {v['plot_auc']:.3f} | {_pct(v['review_share'])} |")
    if m["transfer"]:
        lines += [
            "",
            "## Area never seen by the model or the calibration",
            "",
            "| thresholds | cleared | flagged | review | missed | wrongly flagged |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method, v in m["transfer"].items():
            lines.append(
                f"| {method} | {_pct(v['clear_share'])} | {_pct(v['flag_share'])} | {_pct(v['review_share'])} "
                f"| {_pct(v['false_clear_rate'])} ({v['false_clear']} of {v['n_deforested']}) | {_pct(v['false_flag_rate'])} |"
            )
    lines += ["", "## Label efficiency", "", "| labelled pixels | seeds | plot AUC | review share (min to max) |", "|---:|---:|---:|---|"]
    curve = pd.DataFrame(m["label_efficiency"]).groupby("labelled_pixels")
    for n, part in curve:
        share = part.review_share
        spread = f"{_pct(share.mean())} ({_pct(share.min())} to {_pct(share.max())})"
        lines.append(f"| {n:,} | {len(part)} | {part.plot_auc.mean():.3f} | {spread} |")
    err = m["error_analysis"]
    lines += [
        "",
        "## Error analysis",
        "",
        f"Wrongly flagged test plots: {err['false_flags']}, of which {err['false_flags_with_some_loss']} "
        f"have some reference loss below the 0.5 ha line (median {err['false_flags_median_loss_px']:.0f} pixels).",
        "",
        "| main loss year | deforested test plots | cleared | review | flagged |",
        "|---|---:|---:|---:|---:|",
    ]
    for year, v in err["by_loss_year"].items():
        lines.append(f"| {year} | {v['n']} | {_pct(v['clear'])} | {_pct(v['review'])} | {_pct(v['flag'])} |")
    if "precedents" in m:
        p = m["precedents"]
        lines += [
            "",
            "## Precedents",
            "",
            f"Vote of the {p['k']} most similar calibration plots, on {p['n_queries']:,} test plots: AUC {p['vote_auc']:.3f}, "
            f"majority agrees with the reference on {_pct(p['majority_accuracy'])}. "
            f"Inside the review zone ({p['review_n']} plots) the majority is right on {_pct(p['review_majority_accuracy'])}.",
        ]
    lines += ["", f"Runtime of the experiment on cached data: {m['runtime_s']:.0f} s."]
    return "\n".join(lines) + "\n"
