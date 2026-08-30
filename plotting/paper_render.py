"""Re-render headline figures at ACL-paper size from saved data.

Reads intermediate data files (.json / .jsonl / .npz) produced by the
canonical plotters and re-renders compact, paper-sized versions:
  - Vector PDF  -> paper/figures/<dst>.pdf      (what main.tex includes)
  - Raster PNG  -> results/<...>/_paper.png     (preview alongside talk versions)

The original 12"x4.8" PNGs in results/ are NOT modified.

Run:
    python interpretability/paper_render.py
    python interpretability/paper_render.py --only template_sim
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ACL widths in inches (acl.sty: \textwidth = 6.30in, \columnwidth = 3.04in).
DOUBLE_COL_W = 6.30
SINGLE_COL_W = 3.04
TICK_LS = 5.0    # axis tick label size
PANEL_H = 1.6    # single-column panel height, in inches
HALF_DOUBLE_W = DOUBLE_COL_W / 2.0  # one panel inside a figure* subfigure[0.48\textwidth]

# Canonical stage boundaries (Llama-3.3-70B, 80 layers).
# Grouped by what kind of similarity event they mark, matching the convention
# of the original 12"-wide template_similarity_correct.png:
#   "cross_only"   — only cross-template similarity drops (Stages 1, 2)
#   "within_cross" — within-template drops as well (Stages 3, 4)
CROSS_ONLY_COLOR = "#d6604d"    # red, same as Cross template line
WITHIN_CROSS_COLOR = "#333333"  # near-black
STAGE_GROUPS = [
    (22, "L22", "cross_only"),
    (36, "L36", "cross_only"),
    (40, "L40", "within_cross"),
    (80, "L80", "within_cross"),
]
GROUP_STYLE = {
    "cross_only":   {"color": CROSS_ONLY_COLOR,   "ls": "--"},
    "within_cross": {"color": WITHIN_CROSS_COLOR, "ls": "-."},
}

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
# Figures are written next to the code unless a sibling paper tree exists.
PAPER_FIGS = (REPO.parent / "paper" / "figures"
              if (REPO.parent / "paper" / "figures").is_dir()
              else REPO / "figures")


# ----- Shared aggregation helpers (lifted from run_cot_swap_logit_lens) -----

def _template_means(values, cluster_ids):
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    return unique, np.stack([values[cluster_ids == c].mean(axis=0)
                             for c in unique])


def _bootstrap_ci(values, n_boot=1000, ci=0.95, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.shape[0] == 1:
        return values[0].copy(), values[0].copy()
    boot = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        rows = rng.integers(0, values.shape[0], size=values.shape[0])
        boot[b] = values[rows].mean(axis=0)
    alpha = (1 - ci) / 2
    return np.quantile(boot, alpha, axis=0), np.quantile(boot, 1 - alpha, axis=0)


def _draw_stage_lines(ax, boundaries):
    """Draw the canonical L22/L36/L40/L80 dashed lines.

    `boundaries`: iterable of layer ints to actually draw (subset of the
    canonical four). Color/linestyle picked per-group per [[paper-plot-style]].
    """
    target = set(int(b) for b in boundaries)
    for layer, _name, group in STAGE_GROUPS:
        if layer not in target:
            continue
        style = GROUP_STYLE[group]
        ax.axvline(layer, color=style["color"], lw=0.8, ls=style["ls"],
                   alpha=0.75, zorder=1)


def _stage_xticks(ax, last_layer, boundaries):
    """Bottom x-ticks 0/22/36/40/60/last with boundary ticks bolded+colored."""
    target = set(int(b) for b in boundaries)
    base = [0]
    for b in (22, 36, 40, 60):
        if b < last_layer:
            base.append(b)
    base.append(last_layer)
    ax.set_xticks(base)
    ax.tick_params(axis="x", labelsize=TICK_LS)
    _bold_stage_xticks(ax, target | {80} if last_layer == 80 else target)
    tick_color_map = {22: CROSS_ONLY_COLOR, 36: CROSS_ONLY_COLOR,
                      40: WITHIN_CROSS_COLOR, 80: WITHIN_CROSS_COLOR}
    for tl in ax.get_xticklabels():
        try:
            v = int(tl.get_text())
        except ValueError:
            continue
        if v in target and v in tick_color_map:
            tl.set_color(tick_color_map[v])


def _stage_legend_handles(boundaries):
    """Line2D proxies for the dashed-line groups present in `boundaries`."""
    from matplotlib.lines import Line2D
    target = set(int(b) for b in boundaries)
    handles = []
    has_cross_only = any(b in target for b in (22, 36))
    has_within_cross = any(b in target for b in (40, 80))
    if has_cross_only:
        layers_in_group = [b for b in (22, 36) if b in target]
        lbl = "Cross-only drop (" + ", ".join(f"L{b}" for b in layers_in_group) + ")"
        handles.append(Line2D([0], [0], color=CROSS_ONLY_COLOR, ls="--",
                              lw=1.1, label=lbl))
    if has_within_cross:
        layers_in_group = [b for b in (40, 80) if b in target]
        lbl = "Within + cross drop (" + ", ".join(f"L{b}" for b in layers_in_group) + ")"
        handles.append(Line2D([0], [0], color=WITHIN_CROSS_COLOR, ls="-.",
                              lw=1.1, label=lbl))
    return handles


def _place_legend(ax, handles, ncol=2):
    """Legend below the axes per [[paper-plot-style]] convention."""
    ax.legend(handles=handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, ncol=ncol,
              fontsize=6.5, handlelength=1.6,
              columnspacing=1.0, labelspacing=0.25,
              borderaxespad=0.0)


def setup_rc():
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "axes.xmargin": 0,  # curves touch left/right spines instead of ~5% pad
        "pdf.fonttype": 42,  # TrueType, not Type3 — required by some venues
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _bold_stage_xticks(ax, ticks_to_bold):
    for tl in ax.get_xticklabels():
        try:
            v = int(tl.get_text())
        except ValueError:
            continue
        if v in ticks_to_bold:
            tl.set_fontweight("bold")


def _save(fig, paper_pdf: Path, results_png: Path, tight: bool = True):
    paper_pdf.parent.mkdir(parents=True, exist_ok=True)
    results_png.parent.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(paper_pdf, bbox_inches=bbox)
    fig.savefig(results_png, dpi=300, bbox_inches=bbox)
    plt.close(fig)
    for out in (paper_pdf, results_png):
        try:
            shown = out.relative_to(REPO)
        except ValueError:          # paper tree may live outside the repo
            shown = out
        print(f"  wrote {shown}")


# ---------------------------------------------------------------------------
# Family 1: template_similarity (Fig 1, two-panel figure*)
# ---------------------------------------------------------------------------

def render_template_similarity(mode: str):
    src = RESULTS / "template_similarity" / "llama-3.3-70b-instruct" / \
        "gsm_symbolic" / mode / "template_similarity_correct.json"
    rows = json.loads(src.read_text())
    layers = np.array([r["layer"] for r in rows])
    within = np.array([r["within_mean"] for r in rows])
    cross  = np.array([r["cross_mean"]  for r in rows])
    wlo = np.array([r["within_ci_lo"] for r in rows])
    whi = np.array([r["within_ci_hi"] for r in rows])
    clo = np.array([r["cross_ci_lo"]  for r in rows])
    chi = np.array([r["cross_ci_hi"]  for r in rows])

    fig, ax = plt.subplots(figsize=(HALF_DOUBLE_W, 1.6))
    ax.fill_between(layers, wlo, whi, alpha=0.35, color="#2166ac", linewidth=0)
    ax.fill_between(layers, clo, chi, alpha=0.35, color="#d6604d", linewidth=0)
    line_within, = ax.plot(layers, within, color="#2166ac", lw=1.3,
                           label="Within template")
    line_cross,  = ax.plot(layers, cross,  color="#d6604d", lw=1.3,
                           label="Cross template")

    # Stage layer names are carried by bold colored x-axis ticks at the
    # bottom (no top labels — the bottom ticks already name 22/36/40/80).
    layer_to_idx = {int(l): i for i, l in enumerate(layers)}
    # Per-boundary annotation placement (chosen to avoid overlap with top
    # labels, axis floor, and each other):
    #   dx: horizontal offset in points (positive = right of dot)
    #   ha: text horizontal alignment
    #   dy_within / dy_cross: vertical offsets in points; sign decides
    #     whether the label sits above (+) or below (-) the dot.
    # Annotate both curves at every boundary. Within zigzags ABOVE / BELOW
    # the dot at adjacent boundaries (curve is flat in CoT, so same-side
    # placements collide). Cross at L22 is placed LEFT of dot — RIGHT-side
    # placement would put the bbox on top of the steep cross drop through
    # x=22--30.
    annot_cfg = {
        22: dict(dx_w=+4, ha_w="left",  dy_w=+8,
                 dx_c=-4, ha_c="right", dy_c=-8,
                 annotate_within=True),
        36: dict(dx_w=-4, ha_w="right", dy_w=-8,
                 dx_c=-4, ha_c="right", dy_c=-8,
                 annotate_within=True),
        40: dict(dx_w=+4, ha_w="left",  dy_w=+8,
                 dx_c=+4, ha_c="left",  dy_c=-10,
                 annotate_within=True),
        # L80: within ABOVE dot, cross BELOW dot — placing both in the
        # in-between gap fails in CoT where within=0.41/cross=0.25 are only
        # 0.16 apart. Outside-the-dots placement gives ~0.24 separation.
        80: dict(dx_w=-4, ha_w="right", dy_w=+8,
                 dx_c=-4, ha_c="right", dy_c=-8,
                 annotate_within=True),
    }
    annot_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85)

    for layer, _name, group in STAGE_GROUPS:
        style = GROUP_STYLE[group]
        ax.axvline(layer, color=style["color"], lw=0.8, ls=style["ls"],
                   alpha=0.75, zorder=1)
        # Mark cosine values on both curves at this boundary
        idx = layer_to_idx.get(int(layer))
        if idx is None:
            continue
        cfg = annot_cfg[layer]
        wv = float(within[idx]); cv = float(cross[idx])
        # Scatter dot on whichever curves we annotate
        scatter_xs, scatter_ys, scatter_cs = [layer], [cv], ["#d6604d"]
        if cfg["annotate_within"]:
            scatter_xs.append(layer); scatter_ys.append(wv); scatter_cs.append("#2166ac")
        ax.scatter(scatter_xs, scatter_ys, s=10, color=scatter_cs,
                   edgecolor="white", linewidth=0.5, zorder=5)
        # Cross annotation always
        ax.annotate(f"{cv:.2f}", (layer, cv),
                    xytext=(cfg["dx_c"], cfg["dy_c"]),
                    textcoords="offset points",
                    ha=cfg["ha_c"], va="center",
                    fontsize=5.8, color="#d6604d", fontweight="bold",
                    bbox=annot_bbox, zorder=6)
        # Within annotation
        if cfg["annotate_within"]:
            ax.annotate(f"{wv:.2f}", (layer, wv),
                        xytext=(cfg["dx_w"], cfg["dy_w"]),
                        textcoords="offset points",
                        ha=cfg["ha_w"], va="center",
                        fontsize=5.8, color="#2166ac", fontweight="bold",
                        bbox=annot_bbox, zorder=6)

    # X-ticks: include all four stage boundaries; shrink tick label font so
    # 36 and 40 (4 layers apart) don't collide. Color the stage ticks to
    # match their dashed-line group.
    last_layer = int(layers[-1])
    ax.set_xticks([0, 22, 36, 40, 60, last_layer])
    ax.tick_params(axis="x", labelsize=TICK_LS)
    _bold_stage_xticks(ax, {22, 36, 40, 80})
    tick_color_map = {22: CROSS_ONLY_COLOR, 36: CROSS_ONLY_COLOR,
                      40: WITHIN_CROSS_COLOR, 80: WITHIN_CROSS_COLOR}
    for tl in ax.get_xticklabels():
        try:
            v = int(tl.get_text())
        except ValueError:
            continue
        if v in tick_color_map:
            tl.set_color(tick_color_map[v])

    ax.set_xlabel("Layer", fontsize=7)
    # Shortened from "Mean pairwise cosine similarity" — full version doesn't
    # fit vertically at 2.1" canvas height.
    ax.set_ylabel("Cosine similarity", fontsize=7)
    # Extra headroom (1.06) so above-dot within annotations at L22/L40 fit.
    ax.set_ylim(min(min(clo), min(wlo)) - 0.04, 1.06)
    # Sparse y-ticks (5 vs matplotlib's auto 8) — less busy.
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)

    # Legend in lower-left (matches original 12"-wide figure convention).
    # Four entries: data lines + two boundary-line proxies.
    from matplotlib.lines import Line2D
    cross_only_proxy = Line2D([0], [0], color=CROSS_ONLY_COLOR, ls="--", lw=1.1,
                              label="Cross-only drop")
    within_cross_proxy = Line2D([0], [0], color=WITHIN_CROSS_COLOR, ls="-.", lw=1.1,
                                label="Within + cross drop")
    # Legend placed below the plot so it never crosses the curves or the
    # value annotations. ncol=2 keeps the figure short.
    ax.legend(handles=[line_within, line_cross,
                       cross_only_proxy, within_cross_proxy],
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, ncol=2,
              fontsize=6.8, handlelength=1.8,
              columnspacing=1.4, labelspacing=0.25,
              borderaxespad=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paper_pdf = PAPER_FIGS / "template_similarity" / mode / "template_similarity_correct.pdf"
    results_png = src.with_name("template_similarity_correct_paper.png")
    _save(fig, paper_pdf, results_png)


def render_template_similarity_robust(model: str, mode: str,
                                      boundaries: list[tuple[int, str]]):
    """Render template-similarity for a non-Llama model with model-specific
    stage boundaries (appendix robustness checks).

    `boundaries`: list of (layer, group) where group is "cross_only" or
    "within_cross". Boundaries are model-specific, chosen from where the
    cross-only and within+cross drops actually fall in that model's curve.

    Smaller-model JSONs carry `within_se`/`cross_se` rather than CI bounds;
    we draw ±1.96·SE shading as the analog of the Llama panel's CI band.
    """
    src = RESULTS / "template_similarity" / model / "gsm_symbolic" / mode / \
        "template_similarity_correct.json"
    rows = json.loads(src.read_text())
    layers = np.array([r["layer"] for r in rows])
    within = np.array([r["within_mean"] for r in rows])
    cross  = np.array([r["cross_mean"]  for r in rows])
    wse    = np.array([r["within_se"] for r in rows])
    cse    = np.array([r["cross_se"]  for r in rows])
    wlo, whi = within - 1.96 * wse, within + 1.96 * wse
    clo, chi = cross  - 1.96 * cse, cross  + 1.96 * cse

    fig, ax = plt.subplots(figsize=(HALF_DOUBLE_W, 2.1))
    ax.fill_between(layers, wlo, whi, alpha=0.35, color="#2166ac", linewidth=0)
    ax.fill_between(layers, clo, chi, alpha=0.35, color="#d6604d", linewidth=0)
    line_within, = ax.plot(layers, within, color="#2166ac", lw=1.3,
                           label="Within template")
    line_cross,  = ax.plot(layers, cross,  color="#d6604d", lw=1.3,
                           label="Cross template")

    layer_to_idx = {int(l): i for i, l in enumerate(layers)}
    annot_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85)

    boundary_layers = set()
    for layer, group in boundaries:
        style = GROUP_STYLE[group]
        ax.axvline(layer, color=style["color"], lw=0.8, ls=style["ls"],
                   alpha=0.75, zorder=1)
        idx = layer_to_idx.get(int(layer))
        if idx is None:
            continue
        boundary_layers.add(int(layer))
        wv = float(within[idx]); cv = float(cross[idx])
        ax.scatter([layer, layer], [wv, cv], s=10,
                   color=["#2166ac", "#d6604d"],
                   edgecolor="white", linewidth=0.5, zorder=5)
        # Within annotation above dot, cross below — same logic as the
        # canonical L40/L80 placements (both curves diverging downward).
        ax.annotate(f"{wv:.2f}", (layer, wv),
                    xytext=(+4, +8), textcoords="offset points",
                    ha="left", va="center",
                    fontsize=5.8, color="#2166ac", fontweight="bold",
                    bbox=annot_bbox, zorder=6)
        ax.annotate(f"{cv:.2f}", (layer, cv),
                    xytext=(+4, -8), textcoords="offset points",
                    ha="left", va="center",
                    fontsize=5.8, color="#d6604d", fontweight="bold",
                    bbox=annot_bbox, zorder=6)

    last_layer = int(layers[-1])
    tick_set = sorted({0, last_layer} | boundary_layers)
    ax.set_xticks(tick_set)
    ax.tick_params(axis="x", labelsize=TICK_LS)
    _bold_stage_xticks(ax, boundary_layers)
    group_for = {int(b): g for b, g in boundaries}
    tick_color_map = {b: GROUP_STYLE[group_for[b]]["color"] for b in boundary_layers}
    for tl in ax.get_xticklabels():
        try:
            v = int(tl.get_text())
        except ValueError:
            continue
        if v in tick_color_map:
            tl.set_color(tick_color_map[v])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(min(min(clo), min(wlo)) - 0.04, 1.06)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)

    from matplotlib.lines import Line2D
    legend_handles = [line_within, line_cross]
    groups_used = {g for _, g in boundaries}
    if "cross_only" in groups_used:
        layers_in_group = sorted(b for b, g in boundaries if g == "cross_only")
        lbl = "Cross-only drop (" + ", ".join(f"L{b}" for b in layers_in_group) + ")"
        legend_handles.append(Line2D([0], [0], color=CROSS_ONLY_COLOR, ls="--",
                                     lw=1.1, label=lbl))
    if "within_cross" in groups_used:
        layers_in_group = sorted(b for b, g in boundaries if g == "within_cross")
        lbl = "Within + cross drop (" + ", ".join(f"L{b}" for b in layers_in_group) + ")"
        legend_handles.append(Line2D([0], [0], color=WITHIN_CROSS_COLOR, ls="-.",
                                     lw=1.1, label=lbl))
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, ncol=2,
              fontsize=6.8, handlelength=1.8,
              columnspacing=1.4, labelspacing=0.25,
              borderaxespad=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    paper_pdf = PAPER_FIGS / "template_similarity" / "robust" / model / \
        f"{mode}.pdf"
    results_png = src.with_name("template_similarity_correct_paper.png")
    _save(fig, paper_pdf, results_png)


# ---------------------------------------------------------------------------
# Family 2: presence_probe (Fig 2 pairs headline + Fig 3 plain appendix)
# ---------------------------------------------------------------------------

PRESENCE_COLORS = {
    "entity":           "#c2185b",  # deep magenta
    "op_presence":      "#6a1b9a",  # deep purple
    "narrative_filler": "#616161",  # graphite
    "number":           "#1976d2",  # steel blue (matches supp figure)
    "answer":           "#2e7d32",  # forest green (Stage 4 computation)
}
PRESENCE_LABELS = {
    "entity":           "Entity",
    "op_presence":      "Operator",
    "narrative_filler": "Narrative fillers",
    "number":           "Number tokens",
    "answer":           "Final answer",
}


def _presence_bootstrap_ci(arr, n_boot=1000, ci=0.95, seed=0):
    """Bootstrap CI over per-word probe accuracies. arr shape (L, K)."""
    rng = np.random.default_rng(seed)
    n_layers, K = arr.shape
    if K == 0:
        nan = np.full(n_layers, np.nan)
        return nan, nan, nan
    mean = np.nanmean(arr, axis=1)
    idx = rng.integers(0, K, size=(n_boot, K))
    boot = np.nanmean(arr[:, idx], axis=2)
    alpha = (1 - ci) / 2
    lo = np.nanpercentile(boot, 100 * alpha, axis=1)
    hi = np.nanpercentile(boot, 100 * (1 - alpha), axis=1)
    return mean, lo, hi


def render_presence_probe(dataset_name: str, mode: str, dst_subdir: str,
                          cv_path: str = "template_disjoint",
                          stage_lines: list[int] | None = None,
                          legend_ncol: int | None = None,
                          figsize_h: float = 1.6,
                          subset: str = "correct"):
    """One panel of the presence_probe family.

    dataset_name: 'gsm_symbolic_pairs' (Fig 2) or 'gsm_symbolic' (Fig 3 appendix)
    mode: 'direct' or 'cot'
    dst_subdir: 'pairs' or 'plain' (under paper/figures/presence_probe/)
    cv_path: subdir under <dataset>/, defaults to 'template_disjoint';
        'template_disjoint_unpaired' targets the unpaired-CV-on-pairs-data
        run (more per-word data than literal gsm_symbolic plain).
    stage_lines: which stage-boundary verticals to draw. Defaults to [22]
        (the only stage the abstraction-section presence panels mark);
        pass e.g. [22, 36, 40] to show the full L22/L36/L40 scaffold.
    """
    if stage_lines is None:
        stage_lines = [22]
    src_dir = (RESULTS / "presence_probe" / "llama-3.3-70b-instruct" /
               dataset_name / cv_path / mode / subset)
    npz = np.load(src_dir / "presence_probe.npz")
    # Combine entity_name + entity_other -> entity; function + op_cue -> narrative_filler
    ent = np.concatenate([npz["entity_name_accs"], npz["entity_other_accs"]], axis=1)
    op_presence = npz["op_presence_accs"]
    if op_presence.ndim == 1:
        op_presence = op_presence[:, None]
    narr = np.concatenate([npz["function_accs"], npz["op_cue_accs"]], axis=1)
    number = npz["number_accs"] if "number_accs" in npz.files else None
    if number is not None and number.size == 0:
        number = None
    answer = npz["answer_accs"] if "answer_accs" in npz.files else None
    if answer is not None and answer.size == 0:
        answer = None
    n_layers = ent.shape[0]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(HALF_DOUBLE_W, figsize_h))

    handles = []
    curve_means = {}
    # narrative_filler is computed but not drawn (function-word CV artifact).
    # Per-curve bootstrap CI bands shown for entity/number/answer (K=16-30, a
    # meaningful uncertainty band). op_presence is K=4 so its band reflects
    # word-set heterogeneity more than statistical uncertainty; we still
    # render it for visual consistency.
    curve_set = [("entity", ent), ("op_presence", op_presence)]
    if number is not None:
        curve_set.append(("number", number))
    if answer is not None:
        curve_set.append(("answer", answer))
    for name, arr in curve_set:
        mean, lo, hi = _presence_bootstrap_ci(arr)
        color = PRESENCE_COLORS[name]
        ax.fill_between(layers, lo, hi, alpha=0.20, color=color, linewidth=0)
        ln, = ax.plot(layers, mean, color=color, lw=1.2,
                      label=PRESENCE_LABELS[name])
        handles.append(ln)
        curve_means[name] = mean
    _ = narr  # kept loaded; not drawn

    # Shuffled-label chance baseline — per-layer mean with bootstrap CI band
    # (the band is narrow but real; matches the canonical 12" plotter).
    shuf_pool = []
    for key in ("entity_name_accs_shuffled", "entity_other_accs_shuffled",
                "function_accs_shuffled", "op_cue_accs_shuffled"):
        if key in npz.files and npz[key].size:
            shuf_pool.append(npz[key])
    chance_handle = None
    if shuf_pool:
        sh = np.concatenate(shuf_pool, axis=1)
        sh_mean, _sh_lo, _sh_hi = _presence_bootstrap_ci(sh)
        chance_handle, = ax.plot(
            layers, sh_mean, color="#888888", lw=0.8, ls=":", alpha=0.9,
            label="Shuffled label")

    # Stage boundary markers (subset configurable via `stage_lines`).
    _draw_stage_lines(ax, stage_lines)

    # Value annotations: at L22 (right of marker line) and at pre-L22 peak
    # (left of marker line) for each curve. Color-matched, bbox-masked.
    annot_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85)
    # Per-curve annotation placement at L22. Entity below dot (in the gap to
    # the chance line), op-presence above (plenty of room at the top).
    l22_cfg = {
        "entity":      dict(dy=-8),
        "op_presence": dict(dy=+8),
    }
    peak_cfg = {
        "entity":      dict(dy=+9),
        "op_presence": dict(dy=+9),
    }
    for name, mean in curve_means.items():
        # Skip annotations for curves without a configured placement.
        # Number tokens are drawn as a supporting line (their L40 jump is
        # the visual signal; no inline annotation needed).
        if name not in l22_cfg:
            continue
        color = PRESENCE_COLORS[name]
        # L22 marker
        v22 = float(mean[22])
        ax.scatter([22], [v22], s=10, color=color,
                   edgecolor="white", linewidth=0.5, zorder=5)
        ax.annotate(f"{v22:.2f}", (22, v22),
                    xytext=(4, l22_cfg[name]["dy"]),
                    textcoords="offset points",
                    ha="left", va="center",
                    fontsize=5.8, color=color, fontweight="bold",
                    bbox=annot_bbox, zorder=6)
        # Pre-L22 peak (max value over layers 0..21)
        pre = mean[:22]
        peak_layer = int(np.argmax(pre))
        peak_val = float(pre[peak_layer])
        if peak_layer >= 20 or abs(peak_val - v22) < 0.02:
            continue
        ax.scatter([peak_layer], [peak_val], s=10, color=color,
                   edgecolor="white", linewidth=0.5, zorder=5)
        ax.annotate(f"{peak_val:.2f}", (peak_layer, peak_val),
                    xytext=(0, peak_cfg[name]["dy"]),
                    textcoords="offset points",
                    ha="center", va="center",
                    fontsize=5.8, color=color, fontweight="bold",
                    bbox=annot_bbox, zorder=6)

    # L40 markers (Stage-3 binding boundary). Drawn only when L40 is in the
    # caller's `stage_lines`. Per-curve placement to avoid overlap:
    # entity (0.52) sits left-below its dot to clear the answer (0.53)
    # label which sits right-below; number (~0.77) sits left-below so it
    # doesn't sit on top of the post-L40 curve; operator (~0.83) sits
    # right-above.
    if 40 in set(int(b) for b in stage_lines) and 40 < n_layers:
        l40_cfg = {
            "entity":      dict(dx=-4, dy=-9, ha="right"),
            "op_presence": dict(dx=+4, dy=+9, ha="left"),
            "number":      dict(dx=-4, dy=-9, ha="right"),
            "answer":      dict(dx=+4, dy=-9, ha="left"),
        }
        for name, mean in curve_means.items():
            if name not in l40_cfg:
                continue
            color = PRESENCE_COLORS[name]
            v40 = float(mean[40])
            cfg = l40_cfg[name]
            ax.scatter([40], [v40], s=10, color=color,
                       edgecolor="white", linewidth=0.5, zorder=5)
            ax.annotate(f"{v40:.2f}", (40, v40),
                        xytext=(cfg["dx"], cfg["dy"]),
                        textcoords="offset points",
                        ha=cfg["ha"], va="center",
                        fontsize=5.8, color=color, fontweight="bold",
                        bbox=annot_bbox, zorder=6)

    # L80 markers (Stage-4 output boundary). At the right edge of the plot,
    # so all labels are placed LEFT of their dot (dx=-4, ha="right"). Values
    # at L80: answer ~0.93 (top), operator ~0.72, number ~0.59, entity
    # ~0.52 — answer / operator separated by ~0.21, others bunch between
    # 0.52–0.59 so we offset entity below / number above their dots.
    if 80 in set(int(b) for b in stage_lines) and 80 < n_layers:
        l80_cfg = {
            "entity":      dict(dx=-4, dy=-9, ha="right"),
            "op_presence": dict(dx=-4, dy=+9, ha="right"),
            "number":      dict(dx=-4, dy=+9, ha="right"),
            "answer":      dict(dx=-4, dy=-9, ha="right"),
        }
        for name, mean in curve_means.items():
            if name not in l80_cfg:
                continue
            color = PRESENCE_COLORS[name]
            v80 = float(mean[80])
            cfg = l80_cfg[name]
            ax.scatter([80], [v80], s=10, color=color,
                       edgecolor="white", linewidth=0.5, zorder=5)
            ax.annotate(f"{v80:.2f}", (80, v80),
                        xytext=(cfg["dx"], cfg["dy"]),
                        textcoords="offset points",
                        ha=cfg["ha"], va="center",
                        fontsize=5.8, color=color, fontweight="bold",
                        bbox=annot_bbox, zorder=6)

    last_layer = n_layers - 1
    # X-ticks: include 36/40 only when those stage lines are drawn.
    stage_set = set(int(b) for b in stage_lines)
    base_ticks = [0]
    for t in (22, 36, 40, 60):
        if t == 22 or t in stage_set or t == 60:
            if t < last_layer:
                base_ticks.append(t)
    base_ticks.append(last_layer)
    base_ticks = sorted(set(base_ticks))
    ax.set_xticks(base_ticks)
    ax.tick_params(axis="x", labelsize=TICK_LS)
    _bold_stage_xticks(ax, stage_set)
    tick_color_map = {22: CROSS_ONLY_COLOR, 36: CROSS_ONLY_COLOR,
                      40: WITHIN_CROSS_COLOR, 80: WITHIN_CROSS_COLOR}
    for tl in ax.get_xticklabels():
        try:
            v = int(tl.get_text())
        except ValueError:
            continue
        if v in stage_set and v in tick_color_map:
            tl.set_color(tick_color_map[v])

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Probe accuracy", fontsize=7)
    ax.set_ylim(0.30, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend below the axes (matches template_similarity / Fig 1 convention),
    # so the in-plot region stays clean now that we draw entity + operator +
    # number tokens + final answer + shuffled-chance on the same axes.
    legend_handles = list(handles)
    if chance_handle is not None:
        legend_handles.append(chance_handle)
    ncol = legend_ncol if legend_ncol is not None else len(legend_handles)
    _place_legend(ax, legend_handles, ncol=ncol)

    paper_pdf = PAPER_FIGS / "presence_probe" / dst_subdir / mode / "main.pdf"
    results_png = (src_dir / "main_paper.png")
    _save(fig, paper_pdf, results_png)


def render_presence_probe_robust(model_short: str, stage_lines: list[int],
                                  dataset_name: str = "gsm_symbolic_pairs",
                                  mode: str = "direct",
                                  cv_path: str = "template_disjoint",
                                  correctness: str = "correct"):
    """Llama-style presence-probe panel (entity + op_presence + number + answer
    + shuffled chance) for a non-Llama robustness model, with that model's own
    Surface Stripping / Operand Binding stage boundaries.

    Per-boundary value annotations (entity/op/number/answer) are added at the
    model's L_abs, L_bind, and L_out, matching the Fig 1 panel-b layout.
    """
    src_dir = (RESULTS / "presence_probe" / model_short /
               dataset_name / cv_path / mode / correctness)
    npz_path = src_dir / "presence_probe.npz"
    if not npz_path.exists():
        print(f"  skipped: missing {npz_path}")
        return
    npz = np.load(npz_path)

    ent = np.concatenate([npz["entity_name_accs"], npz["entity_other_accs"]], axis=1)
    op_presence = npz["op_presence_accs"] if "op_presence_accs" in npz.files else np.zeros((0, 0))
    if op_presence.ndim == 1:
        op_presence = op_presence[:, None]
    number = npz["number_accs"] if "number_accs" in npz.files else None
    if number is not None and number.size == 0:
        number = None
    answer = npz["answer_accs"] if "answer_accs" in npz.files else None
    if answer is not None and answer.size == 0:
        answer = None
    n_layers = ent.shape[0]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(HALF_DOUBLE_W, 1.8))
    handles = []
    curve_means = {}
    curve_set = [("entity", ent)]
    if op_presence.size:
        curve_set.append(("op_presence", op_presence))
    if number is not None:
        curve_set.append(("number", number))
    if answer is not None:
        curve_set.append(("answer", answer))
    for name, arr in curve_set:
        mean, lo, hi = _presence_bootstrap_ci(arr)
        color = PRESENCE_COLORS[name]
        ax.fill_between(layers, lo, hi, alpha=0.20, color=color, linewidth=0)
        ln, = ax.plot(layers, mean, color=color, lw=1.2,
                      label=PRESENCE_LABELS[name])
        handles.append(ln)
        curve_means[name] = mean

    # Shuffled-label chance baseline
    shuf_pool = [
        npz[k] for k in (
            "entity_name_accs_shuffled", "entity_other_accs_shuffled",
            "function_accs_shuffled", "op_cue_accs_shuffled",
        )
        if k in npz.files and npz[k].size
    ]
    chance_handle = None
    if shuf_pool:
        sh = np.concatenate(shuf_pool, axis=1)
        sh_mean, _, _ = _presence_bootstrap_ci(sh)
        chance_handle, = ax.plot(
            layers, sh_mean, color="#888888", lw=0.8, ls=":", alpha=0.9,
            label="Shuffled label")

    # Stage boundary lines using the canonical color scheme
    # (cross-only = red dashed for early boundaries; within+cross = dark
    # dash-dot for later ones). Heuristic: stage_lines is sorted; first half
    # cross_only, second half within_cross.
    sorted_lines = sorted(int(b) for b in stage_lines)
    mid = len(sorted_lines) // 2
    for i, layer in enumerate(sorted_lines):
        if layer >= n_layers:
            continue
        group = "cross_only" if i < mid else "within_cross"
        style = GROUP_STYLE[group]
        ax.axvline(layer, color=style["color"], lw=0.8, ls=style["ls"],
                   alpha=0.75, zorder=1)

    # Per-boundary value annotations (mirrors Fig 1 panel b on Llama).
    # Uses the model's own L_abs / L_bind / L_out from sorted_lines.
    annot_bbox = dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="none", alpha=0.85)

    def _mark(layer, cfg):
        if layer >= n_layers:
            return
        for name, opt in cfg.items():
            if name not in curve_means:
                continue
            mean = curve_means[name]
            color = PRESENCE_COLORS[name]
            v = float(mean[layer])
            ax.scatter([layer], [v], s=10, color=color, edgecolor="white",
                       linewidth=0.5, zorder=5)
            ax.annotate(f"{v:.2f}", (layer, v),
                        xytext=(opt["dx"], opt["dy"]),
                        textcoords="offset points",
                        ha=opt["ha"], va="center",
                        fontsize=5.8, color=color, fontweight="bold",
                        bbox=annot_bbox, zorder=6)

    if len(sorted_lines) >= 1:
        # Entity peak before L_abs (highest entity accuracy in [0, L_abs]).
        ent_mean = curve_means.get("entity")
        if ent_mean is not None:
            peak_L = int(np.nanargmax(ent_mean[: sorted_lines[0] + 1]))
            _mark(peak_L, {
                "entity": dict(dx=+4, dy=+9, ha="left"),
            })
        # L_abs: entity drops, op_presence saturates upward.
        _mark(sorted_lines[0], {
            "entity":      dict(dx=+4, dy=-9, ha="left"),
            "op_presence": dict(dx=+4, dy=+9, ha="left"),
        })
    if len(sorted_lines) >= 3:
        # L_bind: number-token jumps; mark all four curves.
        _mark(sorted_lines[2], {
            "entity":      dict(dx=-4, dy=-9, ha="right"),
            "op_presence": dict(dx=+4, dy=+9, ha="left"),
            "number":      dict(dx=-4, dy=-9, ha="right"),
            "answer":      dict(dx=+4, dy=-9, ha="left"),
        })
    if len(sorted_lines) >= 4:
        # L_out: terminal layer; place all labels to the LEFT of the dots
        # (right edge of plot).
        _mark(sorted_lines[3], {
            "entity":      dict(dx=-4, dy=-9, ha="right"),
            "op_presence": dict(dx=-4, dy=+9, ha="right"),
            "number":      dict(dx=-4, dy=+9, ha="right"),
            "answer":      dict(dx=-4, dy=-9, ha="right"),
        })

    # X-ticks at 0, every stage boundary, and last layer
    last_layer = n_layers - 1
    tick_set = sorted(set([0, last_layer] + sorted_lines))
    ax.set_xticks(tick_set)
    ax.tick_params(axis="x", labelsize=TICK_LS)
    _bold_stage_xticks(ax, set(sorted_lines))
    for tl in ax.get_xticklabels():
        try: v = int(tl.get_text())
        except ValueError: continue
        if v in sorted_lines[:mid]:
            tl.set_color(CROSS_ONLY_COLOR)
        elif v in sorted_lines[mid:]:
            tl.set_color(WITHIN_CROSS_COLOR)

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Probe accuracy", fontsize=7)
    ax.set_ylim(0.30, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = list(handles)
    if chance_handle is not None:
        legend_handles.append(chance_handle)
    _place_legend(ax, legend_handles, ncol=len(legend_handles))

    paper_pdf = (PAPER_FIGS / "presence_probe" / "robust" /
                 model_short / f"{correctness}_main.pdf")
    results_png = src_dir / "main_paper_robust.png"
    _save(fig, paper_pdf, results_png)


# ---------------------------------------------------------------------------
# Family 3: cot_swap_patching overlay (Fig 3 Stage-2 headline)
# ---------------------------------------------------------------------------

def _load_cot_swap_effects(path: Path, min_gap=0.5, field="effects",
                           min_pairs_per_template: int = 0):
    """Aggregate cot_swap patching effects from a jsonl into per-layer
    template-bootstrap mean+CI.

    min_pairs_per_template: drop templates with fewer than this many pairs
    (post-gap-filter). Thin templates produce unreliable per-template means
    that contribute equal weight in the template-bootstrap, so a handful of
    1-3-pair templates with outlier means can shift the grand mean by
    several CI widths. Set to 0 (default) to keep all templates.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows
            if float(r.get("denom_source_minus_target", 0.0)) >= min_gap]
    if not rows:
        return None
    if min_pairs_per_template > 0:
        from collections import Counter
        counts = Counter(r["original_id"] for r in rows)
        rows = [r for r in rows
                if counts[r["original_id"]] >= min_pairs_per_template]
        if not rows:
            return None
    effects = np.array([r[field] for r in rows], dtype=float)
    cluster_ids = np.array([r["original_id"] for r in rows])
    finite_layers = np.isfinite(effects).any(axis=0)
    effects = effects[:, finite_layers]
    block_idx = np.arange(finite_layers.shape[0])[finite_layers]
    layers_arr = block_idx + 1   # post-block convention
    tids, eff_tmpl = _template_means(np.nan_to_num(effects), cluster_ids)
    lo, hi = _bootstrap_ci(eff_tmpl, n_boot=1000, ci=0.95)
    return dict(layers=layers_arr, mean=eff_tmpl.mean(axis=0), lo=lo, hi=hi,
                n_templates=len(tids), num_layers=int(finite_layers.shape[0]))


def render_cot_swap_patching():
    base = (RESULTS / "cot_swap_activation_patching" / "llama-3.3-70b-instruct" /
            "p1_vs_padded_symbolic" / "delta0" / "all" / "question_span" / "layer")
    restore = _load_cot_swap_effects(base / "restore.jsonl")
    disrupt = _load_cot_swap_effects(base / "disrupt.jsonl")
    if restore is None or disrupt is None:
        print("  skipped: missing restore or disrupt data")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.1))
    restore_color = "#1b7837"  # green
    disrupt_color = "#762a83"  # purple
    ax.fill_between(restore["layers"], restore["lo"], restore["hi"],
                    color=restore_color, alpha=0.20, linewidth=0)
    ln_r, = ax.plot(restore["layers"], restore["mean"], color=restore_color,
                    lw=1.3, label="Restore (symbolic→p1)")
    ax.fill_between(disrupt["layers"], disrupt["lo"], disrupt["hi"],
                    color=disrupt_color, alpha=0.20, linewidth=0)
    ln_d, = ax.plot(disrupt["layers"], disrupt["mean"], color=disrupt_color,
                    lw=1.3, label="Disrupt (p1→symbolic)")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")

    _draw_stage_lines(ax, [22, 36])
    last_layer = restore["num_layers"]
    _stage_xticks(ax, last_layer, [22, 36])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized patching effect")
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Stage-marker dashed lines kept in plot but not in legend — the colored
    # bottom x-ticks (22/36 in red) name them visually.
    _place_legend(ax, [ln_r, ln_d], ncol=2)

    paper_pdf = PAPER_FIGS / "cot_swap_patching" / "p1_vs_padded_symbolic_delta0_overlay.pdf"
    results_png = base / "overlay_paper.png"
    _save(fig, paper_pdf, results_png)


def render_p1_stage2_panel_a(metric: str = "logp"):
    """Fig 3 Panel A: question-span patching effect.

    `metric`:
      "logp" (default, paper Fig 3a) — single residual trace of the
        normalized patching effect on log P(P1's gold answer). Matches
        Fig 4a's single-trace style.
      "emit" (kept for swap-back) — two-curve answer-emission rate (source
        padded-Sym's answer vs target P1's answer) over free-CoT
        continuation.

    Both modes read the same `padded_to_p1_l*.jsonl` data; only the metric
    column differs. `metric="emit"` saves to `p1_stage2_panel_a_emit.pdf`
    so the logP version stays at the canonical `p1_stage2_panel_a.pdf`.
    """
    import os, math
    rel = ("p1_padded_delta0_patching/llama-3.3-70b-instruct/"
           "p1_vs_padded_aligned_delta0_tfm/cot_boundary/question_span/layer")
    src_dir = str(RESULTS / rel)
    if not os.path.isdir(src_dir):
        print("  skipped: panel_a data dir missing")
        return
    total_layers = 80
    layers80 = np.arange(total_layers) + 1
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    ax.axvspan(22, 36, color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)

    if metric == "logp":
        # Single residual trace of normalized_gold_logprob_effects (Fig 4a style).
        succ_by_layer, cls_by_layer = {}, {}
        for f in sorted(os.listdir(src_dir)):
            if not f.startswith("padded_to_p1_l") or not f.endswith(".jsonl"):
                continue
            L = int(f.replace(".jsonl", "").rsplit("_l", 1)[1])
            vals, cids = [], []
            for line in open(os.path.join(src_dir, f)):
                r = json.loads(line)
                # Canonical filter: |source-target gold-logprob gap| > 0.5
                if abs(float(r.get("source_minus_target_gold_logprob", 0))) <= 0.5:
                    continue
                ne = r["normalized_gold_logprob_effects"][L]
                if isinstance(ne, list) and ne:
                    ne = ne[0]
                if ne is None or (isinstance(ne, float) and math.isnan(ne)):
                    continue
                vals.append(float(ne))
                cids.append(r["original_id"])
            if vals:
                succ_by_layer[L] = np.array(vals)
                cls_by_layer[L] = np.array(cids)

        mean = np.full(total_layers, np.nan)
        lo = np.full(total_layers, np.nan)
        hi = np.full(total_layers, np.nan)
        for L, vals in succ_by_layer.items():
            _, t = _template_means(np.nan_to_num(vals.reshape(-1, 1)), cls_by_layer[L])
            mean[L] = float(np.nanmean(t))
            l_, h_ = _bootstrap_ci(t, n_boot=500, ci=0.95)
            lo[L] = float(np.atleast_1d(l_)[0])
            hi[L] = float(np.atleast_1d(h_)[0])

        fin = np.isfinite(mean)
        ax.fill_between(layers80, lo, hi, color="#333333", alpha=0.20,
                        linewidth=0, where=fin)
        ln, = ax.plot(layers80, mean, color="#333333", lw=1.3, label="residual")
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
        ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")
        legend_handles = [ln]
        ylim = (-0.05, 1.15)
        yticks = [0.0, 0.5, 1.0]
        ylabel = "Patching effect"
        pdf_name = "p1_stage2_panel_a.pdf"
    else:  # metric == "emit"
        succ_by_layer = {}
        clusters_by_layer = {}
        for f in sorted(os.listdir(src_dir)):
            if not f.startswith("padded_to_p1_l") or not f.endswith(".jsonl"):
                continue
            L = int(f.replace(".jsonl", "").rsplit("_l", 1)[1])
            succs, oids = [], []
            for line in open(os.path.join(src_dir, f)):
                r = json.loads(line)
                s = r["patched_final_answer_success"][L]
                if isinstance(s, list) and s:
                    s = s[0]
                succs.append(float(s) if s is not None and not (isinstance(s, float) and math.isnan(s))
                             else float("nan"))
                oids.append(r["original_id"])
            succ_by_layer[L] = np.array(succs)
            clusters_by_layer[L] = np.array(oids)

        def curve(transform=None):
            mean = np.full(total_layers, np.nan)
            lo = np.full(total_layers, np.nan)
            hi = np.full(total_layers, np.nan)
            for L, vals in succ_by_layer.items():
                if vals.size == 0:
                    continue
                v = transform(vals) if transform else vals
                cls = clusters_by_layer[L]
                _, t = _template_means(v.reshape(-1, 1), cls)
                mean[L] = float(np.nanmean(t))
                l_, h_ = _bootstrap_ci(t, n_boot=500, ci=0.95)
                lo[L] = float(np.atleast_1d(l_)[0])
                hi[L] = float(np.atleast_1d(h_)[0])
            return mean, lo, hi

        m_s, lo_s, hi_s = curve(transform=lambda v: 1.0 - v)
        m_t, lo_t, hi_t = curve()
        color_s, color_t = "#b35806", "#2166ac"
        ax.fill_between(layers80, lo_s, hi_s, color=color_s, alpha=0.30,
                        linewidth=0, where=np.isfinite(m_s))
        ax.fill_between(layers80, lo_t, hi_t, color=color_t, alpha=0.30,
                        linewidth=0, where=np.isfinite(m_t))
        ln_s, = ax.plot(layers80, m_s, color=color_s, lw=1.3,
                        label="padded-Symbolic answer")
        ln_t, = ax.plot(layers80, m_t, color=color_t, lw=1.3, label="P1 answer")
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
        legend_handles = [ln_s, ln_t]
        ylim = (-0.04, 1.06)
        yticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        ylabel = "Answer emit rate"
        pdf_name = "p1_stage2_panel_a_emit.pdf"

    _draw_stage_lines(ax, [22, 36])
    _stage_xticks(ax, total_layers, [22, 36])

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, legend_handles, ncol=len(legend_handles))

    paper_pdf = PAPER_FIGS / "cot_swap_patching" / pdf_name
    results_png = (RESULTS / "cot_swap_activation_patching" /
                   "llama-3.3-70b-instruct" /
                   "p1_vs_padded_symbolic_cot-symbolic_aligned" /
                   "gsm_padded_symbolic_p1_aligned" / "delta0" / "all" /
                   pdf_name.replace(".pdf", ".png"))
    _save(fig, paper_pdf, results_png, tight=False)


def render_p1_stage2_panel_c():
    """Fig 2 Panel B: cot_end commit-readiness component decomposition.

    Residual / attention / MLP curves at the end of the injected CoT
    (normalized effect on log P(####)). Dimensions match Fig 3 panels for
    2-panel figure* layout at 0.48*\\textwidth subfigures."""
    base = (RESULTS / "cot_swap_activation_patching" / "llama-3.3-70b-instruct" /
            "p1_vs_padded_symbolic" / "delta0" / "all" / "cot_end")
    scopes = [
        ("layer",       "residual",  "#333333"),
        ("attn_output", "attention", "#1b7837"),
        ("mlp",         "MLP",       "#762a83"),
    ]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    ax.axvspan(22, 36, color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)

    handles = []
    num_layers = None
    for scope, label, color in scopes:
        data = _load_cot_swap_effects(base / scope / "restore.jsonl",
                                      field="normalized_effects")
        if data is None:
            print(f"  skipped: missing {scope}/restore")
            continue
        num_layers = data["num_layers"]
        layers = data["layers"]
        ax.fill_between(layers, data["lo"], data["hi"],
                        color=color, alpha=0.30, linewidth=0)
        ln, = ax.plot(layers, data["mean"], color=color, lw=1.3, label=label)
        handles.append(ln)
    if not handles:
        print("  skipped: no cot_end data")
        plt.close(fig)
        return

    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, num_layers or 80, [22, 36, 40])

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.set_ylim(-0.12, 1.14)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, handles, ncol=len(handles))

    paper_pdf = PAPER_FIGS / "cot_swap_patching" / "p1_stage2_panel_c.pdf"
    results_png = (RESULTS / "cot_swap_activation_patching" /
                   "llama-3.3-70b-instruct" /
                   "p1_vs_padded_symbolic_cot-symbolic_aligned" /
                   "gsm_padded_symbolic_p1_aligned" / "delta0" / "all" /
                   "p1_stage2_panel_c.png")
    _save(fig, paper_pdf, results_png, tight=False)


def _robust_stage_decorate(ax, stage_lines, last_layer):
    """Draw boundary axvlines + custom xticks for non-Llama models.

    Splits stage_lines in half: first half = cross_only (Surface Stripping),
    second half = within_cross (Operand Binding / Computation), matching the
    color scheme used in render_presence_probe_robust.
    """
    sorted_lines = sorted(int(b) for b in stage_lines)
    mid = len(sorted_lines) // 2
    for i, layer in enumerate(sorted_lines):
        if layer > last_layer:
            continue
        group = "cross_only" if i < mid else "within_cross"
        style = GROUP_STYLE[group]
        ax.axvline(layer, color=style["color"], lw=0.8, ls=style["ls"],
                   alpha=0.75, zorder=1)
    tick_set = sorted(set([0, last_layer] + [b for b in sorted_lines if b <= last_layer]))
    ax.set_xticks(tick_set)
    ax.tick_params(axis="x", labelsize=TICK_LS)
    _bold_stage_xticks(ax, set(sorted_lines))
    for tl in ax.get_xticklabels():
        try: v = int(tl.get_text())
        except ValueError: continue
        if v in sorted_lines[:mid]:
            tl.set_color(CROSS_ONLY_COLOR)
        elif v in sorted_lines[mid:]:
            tl.set_color(WITHIN_CROSS_COLOR)


def render_stage2_robust(model_short: str, stage_lines: list[int]):
    """Stage-2 counterpart for robustness models (companion to Fig 3).

    Two panels: (a) question_span restore+disrupt overlay on log P(####)
    normalized effect; (b) cot_end commit-readiness decomposition (full
    residual / attn / mlp). stage_lines = [L_abs, L_form] in model layers.
    """
    qs_base = (RESULTS / "cot_swap_activation_patching" / model_short /
               "p1_vs_padded_symbolic" / "delta0" / "all" /
               "question_span" / "layer")
    ce_base = (RESULTS / "cot_swap_activation_patching" / model_short /
               "p1_vs_padded_symbolic_cot-symbolic_aligned" /
               "gsm_padded_symbolic_p1_aligned" / "delta0" / "all" /
               "cot_end")

    # Panel (a): question_span normalized patching effect on log P(P1's gold)
    # via the noop_patching (p1_padded_delta0_patching) pipeline. Single
    # residual trace, Fig 4a-style — matches new main Fig 3a.
    import math, os
    qs_src = (RESULTS / "p1_padded_delta0_patching" / model_short /
              "p1_vs_padded_aligned_delta0_tfm" / "cot_boundary" /
              "question_span" / "layer")
    if not qs_src.is_dir():
        print(f"  [{model_short}] panel A skipped: {qs_src} not found")
    else:
        vals_by_layer = {}
        for f in sorted(os.listdir(qs_src)):
            if not (f.startswith("padded_to_p1_l") and f.endswith(".jsonl")):
                continue
            L = int(f.replace(".jsonl", "").rsplit("_l", 1)[1])
            vals = []
            for line in open(qs_src / f):
                r = json.loads(line)
                if abs(float(r.get("source_minus_target_gold_logprob", 0))) <= 0.5:
                    continue
                ne = r["normalized_gold_logprob_effects"][L]
                if isinstance(ne, list) and ne:
                    ne = ne[0]
                if ne is None or (isinstance(ne, float) and math.isnan(ne)):
                    continue
                vals.append(float(ne))
            if vals:
                vals_by_layer[L] = np.array(vals, dtype=float)

        if not vals_by_layer:
            print(f"  [{model_short}] panel A skipped: no rows after gap filter")
        else:
            total_layers = max(vals_by_layer) + 1
            layers = np.arange(total_layers) + 1
            mean = np.full(total_layers, np.nan)
            lo   = np.full(total_layers, np.nan)
            hi   = np.full(total_layers, np.nan)
            rng = np.random.default_rng(0)
            n_boot = 1000
            min_pairs_at_layer = 20
            for L, vals in vals_by_layer.items():
                if vals.size < min_pairs_at_layer:
                    continue
                mean[L] = float(vals.mean())
                boot = np.empty(n_boot, dtype=float)
                for b in range(n_boot):
                    idx = rng.integers(0, vals.size, size=vals.size)
                    boot[b] = vals[idx].mean()
                lo[L] = float(np.quantile(boot, 0.025))
                hi[L] = float(np.quantile(boot, 0.975))

            fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                                   constrained_layout=True)
            sorted_lines = sorted(int(b) for b in stage_lines)
            if len(sorted_lines) >= 2:
                ax.axvspan(sorted_lines[0], sorted_lines[-1],
                           color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
            # Drop NaN/under-sampled layers entirely so the line stays
            # continuous (otherwise a single bad layer breaks the trace
            # into visually-disconnected segments).
            fin = np.isfinite(mean)
            layers_v = layers[fin]
            ax.fill_between(layers_v, lo[fin], hi[fin], color="#333333",
                            alpha=0.20, linewidth=0)
            ln, = ax.plot(layers_v, mean[fin], color="#333333", lw=1.3,
                          label="residual")
            ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
            ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")
            _robust_stage_decorate(ax, stage_lines, total_layers)
            ax.set_xlabel("Layer", fontsize=7)
            ax.set_ylabel("Patching effect", fontsize=7)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.set_ylim(-0.05, 1.15)
            ax.tick_params(axis="y", labelsize=TICK_LS)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            _place_legend(ax, [ln], ncol=1)
            paper_pdf = (PAPER_FIGS / "cot_swap_patching" / "robust" /
                         model_short / "p1_stage2_panel_a.pdf")
            results_png = qs_src / "panel_a_robust.png"
            _save(fig, paper_pdf, results_png)

    # Panel (b): cot_end commit-readiness decomposition
    scopes = [
        ("layer",       "residual",  "#333333"),
        ("attn_output", "attention", "#1b7837"),
        ("mlp",         "MLP",       "#762a83"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    sorted_lines = sorted(int(b) for b in stage_lines)
    if len(sorted_lines) >= 2:
        ax.axvspan(sorted_lines[0], sorted_lines[-1],
                   color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)

    handles = []
    num_layers = None
    for scope, label, color in scopes:
        data = _load_cot_swap_effects(ce_base / scope / "restore.jsonl",
                                      field="normalized_effects",
                                      min_pairs_per_template=5)
        if data is None:
            print(f"  [{model_short}] panel B: skipped scope={scope}")
            continue
        num_layers = data["num_layers"]
        layers = data["layers"]
        ax.fill_between(layers, data["lo"], data["hi"],
                        color=color, alpha=0.30, linewidth=0)
        ln, = ax.plot(layers, data["mean"], color=color, lw=1.3,
                      label=label)
        handles.append(ln)
        print(f"  [{model_short}] panel B {scope}: n_rows={data.get('n_rows', '?')}, "
              f"n_templates={data['n_templates']}")
    if not handles:
        print(f"  [{model_short}] panel B skipped: no cot_end data")
        plt.close(fig)
        return
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    _robust_stage_decorate(ax, stage_lines, num_layers)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.set_ylim(-0.12, 1.14)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _place_legend(ax, handles, ncol=len(handles))
    paper_pdf = (PAPER_FIGS / "cot_swap_patching" / "robust" /
                 model_short / "p1_stage2_panel_c.pdf")
    results_png = ce_base / "panel_c_robust.png"
    _save(fig, paper_pdf, results_png, tight=False)


def _load_noop_patching_curve(base, stem, min_abs_gap=0.5,
                              min_pairs_at_layer=20,
                              aggregation="pair",
                              min_pairs_per_template=1,
                              clip_range=(-1.0, 2.0)):
    """Per-layer aggregate of `normalized_effects` from noop_patching layer
    jsonls. Filters: |source_target_gap| > min_abs_gap (finite) AND
    non-NaN normalized_effect. Layers with fewer than
    `min_pairs_at_layer` valid pairs are masked to NaN.

    aggregation:
      "pair"     — mean of per-pair effects; bootstrap CI over pairs. Good
                   when per-pair variance is low and templates are few.
      "template" — mean of per-template means (each template's per-layer
                   value is the mean of its pairs); bootstrap CI over
                   templates. Use when per-pair variance is high but
                   within-template correlation is also high (CIs become
                   driven by template count rather than the per-pair noise).
    min_pairs_per_template (template mode): drop templates with fewer
                   pairs than this at the given layer."""
    import os, math
    pairs_by_layer = {}   # layer -> list of (effect, original_id)
    if not base.is_dir():
        return None
    for f in sorted(os.listdir(base)):
        if not (f.startswith(f"{stem}_l") and f.endswith(".jsonl")):
            continue
        L = int(f.replace(".jsonl", "").rsplit("_l", 1)[1])
        rows = []
        for line in open(base / f):
            r = json.loads(line)
            g = r.get("source_target_gap")
            if g is None:
                continue
            g = float(g)
            if not math.isfinite(g) or abs(g) <= min_abs_gap:
                continue
            e = r["normalized_effects"][L]
            if isinstance(e, list) and e:
                e = e[0]
            if e is None:
                continue
            ev = float(e)
            if not math.isfinite(ev):
                continue
            # Clip extreme outliers (numerical artifacts from tiny
            # source-target gap denominators) to a sensible range.
            # normalized_effect should be ~[0, 1]; values outside
            # [-1, 2] are nearly always denominator artifacts.
            if clip_range is not None:
                ev = max(clip_range[0], min(clip_range[1], ev))
            rows.append((ev, int(r["original_id"])))
        pairs_by_layer[L] = rows
    if not pairs_by_layer:
        return None
    total_layers = max(pairs_by_layer) + 1
    mean = np.full(total_layers, np.nan)
    lo   = np.full(total_layers, np.nan)
    hi   = np.full(total_layers, np.nan)
    rng = np.random.default_rng(0)
    n_boot = 1000
    n_pairs_per_layer = np.zeros(total_layers, dtype=int)
    for L, rows in pairs_by_layer.items():
        if not rows:
            continue
        effs = np.array([e for e, _ in rows], dtype=float)
        oids = np.array([o for _, o in rows])
        n_pairs_per_layer[L] = effs.size
        if effs.size < min_pairs_at_layer:
            continue
        if aggregation == "template":
            from collections import defaultdict
            by_tpl = defaultdict(list)
            for e, o in rows:
                by_tpl[o].append(e)
            by_tpl = {t: v for t, v in by_tpl.items()
                      if len(v) >= min_pairs_per_template}
            if not by_tpl:
                continue
            tpl_means = np.array([np.mean(v) for v in by_tpl.values()])
            if tpl_means.size < 2:
                continue
            mean[L] = float(tpl_means.mean())
            boot = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                idx = rng.integers(0, tpl_means.size, size=tpl_means.size)
                boot[b] = tpl_means[idx].mean()
            lo[L] = float(np.quantile(boot, 0.025))
            hi[L] = float(np.quantile(boot, 0.975))
        else:
            mean[L] = float(effs.mean())
            boot = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                idx = rng.integers(0, effs.size, size=effs.size)
                boot[b] = effs[idx].mean()
            lo[L] = float(np.quantile(boot, 0.025))
            hi[L] = float(np.quantile(boot, 0.975))
    return dict(mean=mean, lo=lo, hi=hi, num_layers=total_layers,
                n_pairs_per_layer=n_pairs_per_layer)


def render_noop_stage2_robust(model_short: str, stage_lines: list[int],
                              pair_dataset: str = "filler_df_vs_noop_clean_tfm",
                              aggregation: str = "pair",
                              min_pairs_per_template: int = 1):
    """Fig 5a counterpart for robustness models — restore direction only,
    Fig 4a single-trace style. Default pair_dataset = filler_df_vs_noop_clean_tfm
    (digit-free Filler-DF). For the digit-bearing filler_vs_noop_tfm
    variant the per-pair variance is high (bimodal effects across the
    harder pair pool), so pass aggregation="template" to bootstrap over
    per-template means instead — tighter CI bands at the cost of effective
    sample size.
    """
    base = (RESULTS / "noop_patching" / model_short / pair_dataset /
            "cot_boundary" / "question_span" / "layer")
    # restore direction in noop_patching naming: correct_to_wrong
    # (source=filler-correct, target=noop-wrong → patch filler into noop)
    restore = _load_noop_patching_curve(base, "correct_to_wrong",
                                         aggregation=aggregation,
                                         min_pairs_per_template=min_pairs_per_template)
    if restore is None:
        print(f"  [{model_short}] noop_stage2 skipped: {base} has no data")
        return
    num_layers = restore["num_layers"]
    layers = np.arange(num_layers) + 1

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    sorted_lines = sorted(int(b) for b in stage_lines)
    if len(sorted_lines) >= 2:
        ax.axvspan(sorted_lines[0], sorted_lines[-1],
                   color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    # Drop under-sampled / NaN layers entirely so matplotlib draws a
    # continuous trace across them (instead of jagged 0 dips from
    # 1-pair samples that the old loader exposed).
    fin = np.isfinite(restore["mean"])
    layers_v = layers[fin]
    ax.fill_between(layers_v, restore["lo"][fin], restore["hi"][fin],
                    color="#333333", alpha=0.20, linewidth=0)
    ln, = ax.plot(layers_v, restore["mean"][fin], color="#333333", lw=1.3,
                  label="residual")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")
    _robust_stage_decorate(ax, stage_lines, num_layers)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _place_legend(ax, [ln], ncol=1)
    paper_pdf = (PAPER_FIGS / "cot_swap_patching" / "robust" /
                 model_short / "noop_stage2_panel_a.pdf")
    results_png = base / "noop_stage2_robust.png"
    _save(fig, paper_pdf, results_png)


def render_noop_direct_decomp_robust(model_short: str, stage_lines: list[int],
                                     pair_dataset: str = "filler_df_vs_noop_clean_tfm"):
    """Fig 5b counterpart for robustness models — direct prompt-end
    decomposition (residual / attention / MLP), restore direction. Mirrors
    the styling of render_noop_direct_scope_compare for the Llama Fig 5b.
    """
    base = (RESULTS / "noop_patching" / model_short / pair_dataset / "direct")
    scopes = [
        ("layer",       "residual",  "#333333"),
        ("attn_output", "attention", "#1b7837"),
        ("mlp",         "MLP",       "#762a83"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    sorted_lines = sorted(int(b) for b in stage_lines)
    if len(sorted_lines) >= 2:
        ax.axvspan(sorted_lines[0], sorted_lines[-1],
                   color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    handles = []
    num_layers = None
    for scope, label, color in scopes:
        # Restore direction in noop_patching naming: correct_to_wrong
        # (source=filler-correct, target=noop-wrong → patch filler into noop).
        # Matches the panel-A Fig 15 convention.
        data = _load_noop_direct_curve(base / scope, "correct_to_wrong")
        if data is None:
            print(f"  [{model_short}] direct decomp skipped scope={scope}")
            continue
        num_layers = data["num_layers"]
        # Drop NaN points so matplotlib draws a continuous line across
        # under-sampled layers (the gaps come from layer-shards that
        # haven't completed yet, not from a real zero-effect signal).
        m = np.isfinite(data["mean"])
        if not m.any():
            print(f"  [{model_short}] direct decomp: all-NaN scope={scope}")
            continue
        layers_v = data["layers"][m]
        mean_v = data["mean"][m]
        lo_v = data["lo"][m]
        hi_v = data["hi"][m]
        ax.fill_between(layers_v, lo_v, hi_v,
                        color=color, alpha=0.18, linewidth=0)
        ln, = ax.plot(layers_v, mean_v, color=color, lw=1.3, label=label)
        handles.append(ln)
    if not handles:
        print(f"  [{model_short}] direct decomp: no data")
        plt.close(fig)
        return
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")
    _robust_stage_decorate(ax, stage_lines, num_layers)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylim(-0.05, 1.15)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _place_legend(ax, handles, ncol=len(handles))
    paper_pdf = (PAPER_FIGS / "cot_swap_patching" / "robust" /
                 model_short / "noop_stage2_panel_b.pdf")
    results_png = base / "direct_decomp_robust.png"
    _save(fig, paper_pdf, results_png)


# ---------------------------------------------------------------------------
# Family 4: cot_swap_lens (Fig 4 Stage-2 correlational)
# ---------------------------------------------------------------------------

def render_cot_swap_lens():
    path = (RESULTS / "cot_swap_logit_lens" / "llama-3.3-70b-instruct" /
            "p1_vs_symbolic" / "all" / "all" / "readiness_logit_lens.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]
    src = np.array([r["source_hash_logprobs"] for r in rows])
    tgt = np.array([r["target_hash_logprobs"] for r in rows])
    cids = np.array([r["original_id"] for r in rows])
    valid = np.isfinite(src).all(axis=1) & np.isfinite(tgt).all(axis=1)
    src, tgt, cids = src[valid], tgt[valid], cids[valid]
    n, num_layers = src.shape
    layers = np.arange(1, num_layers + 1)

    _, src_tmpl = _template_means(src, cids)
    _, tgt_tmpl = _template_means(tgt, cids)
    src_lo, src_hi = _bootstrap_ci(src_tmpl)
    tgt_lo, tgt_hi = _bootstrap_ci(tgt_tmpl)

    src_label = rows[0].get("source_label", "source")
    tgt_label = rows[0].get("target_label", "target")

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.1))
    src_color = "#2166ac"
    tgt_color = "#d6604d"
    ax.fill_between(layers, src_lo, src_hi, color=src_color, alpha=0.25, linewidth=0)
    ax.fill_between(layers, tgt_lo, tgt_hi, color=tgt_color, alpha=0.25, linewidth=0)
    ln_s, = ax.plot(layers, src_tmpl.mean(axis=0), color=src_color, lw=1.3,
                    label=src_label)
    ln_t, = ax.plot(layers, tgt_tmpl.mean(axis=0), color=tgt_color, lw=1.3,
                    label=tgt_label)

    _draw_stage_lines(ax, [22, 36])
    _stage_xticks(ax, num_layers, [22, 36])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean log P(answer marker)")
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, [ln_s, ln_t], ncol=2)

    paper_pdf = PAPER_FIGS / "cot_swap_lens" / "p1_vs_symbolic_readiness_logit_lens.pdf"
    results_png = path.with_name("readiness_logit_lens_paper.png")
    _save(fig, paper_pdf, results_png)


# ---------------------------------------------------------------------------
# Family 5: within_template_patching (Fig 5 cliff + Fig 6 decomp)
# ---------------------------------------------------------------------------

def _load_per_layer_effects(layer_dir: Path,
                            metric_key: str = "normalized_gold_logprob_effects",
                            min_abs_gap: float = 0.5):
    """cot_boundary stores one jsonl per patched layer. Aggregate to a curve.

    Default metric is `normalized_gold_logprob_effects` — the continuous
    log-prob-recovery measure used by the canonical summary plot. The
    `effects` field is a 0/1 success indicator (much noisier mean).

    Applies `|source_target_gap| > min_abs_gap` filter to drop rows where
    the normalization denominator is too small (which can blow up the
    effect by orders of magnitude). Mirrors
    run_noop_activation_patching._filter_by_cot_boundary_gap.
    """
    files = sorted(layer_dir.glob("source_to_target_l*.jsonl"))
    if not files:
        return None
    # First pass to determine num_layers
    sample = json.loads(open(files[0]).readline())
    num_layers = len(sample[metric_key])
    rows_per_layer = {}
    cluster_per_layer = {}
    for f in files:
        try:
            L = int(f.stem.rsplit("_l", 1)[-1])
        except ValueError:
            continue
        vals, cids = [], []
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if abs(float(r.get("source_target_gap", 0.0))) <= min_abs_gap:
                continue
            arr = r.get(metric_key)
            if arr is None or L >= len(arr) or arr[L] is None:
                continue
            v = arr[L][0] if isinstance(arr[L], list) else arr[L]
            if v is None or not np.isfinite(v):
                continue
            vals.append(float(v))
            cids.append(int(r["original_id"]))
        if vals:
            rows_per_layer[L] = np.array(vals)
            cluster_per_layer[L] = np.array(cids)

    if not rows_per_layer:
        return None
    layers_arr = np.array(sorted(rows_per_layer))
    means = np.empty(len(layers_arr))
    los   = np.empty(len(layers_arr))
    his   = np.empty(len(layers_arr))
    for i, L in enumerate(layers_arr):
        vals = rows_per_layer[L]
        cids = cluster_per_layer[L]
        _, tmpl = _template_means(vals[:, None], cids)
        means[i] = tmpl.mean()
        lo, hi = _bootstrap_ci(tmpl)
        los[i] = lo[0]; his[i] = hi[0]
    return dict(layers=layers_arr + 1, mean=means, lo=los, hi=his,
                num_layers=num_layers)


def render_within_template_patching(scope: str = "question_span"):
    """Headline: cot_boundary cliff.

    scope:
      "question_span"               — patch every token in the question span
                                       at each layer (original, broad).
      "value_tokens_position_exact" — patch only at value-token positions,
                                       requiring source/target value-token
                                       positions match (surgical).
    """
    layer_dir = (RESULTS / "gsm_sym_within_template_patching" /
                 "llama-3.3-70b-instruct" / "cot_boundary" / scope / "layer")
    data = _load_per_layer_effects(layer_dir)
    if data is None:
        print(f"  skipped: no cot_boundary/{scope} data")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    ax.axvspan(36, 40, color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    color = "#333333"
    ax.fill_between(data["layers"], data["lo"], data["hi"],
                    color=color, alpha=0.20, linewidth=0)
    ln, = ax.plot(data["layers"], data["mean"], color=color, lw=1.3,
                  label="residual")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, data["num_layers"], [22, 36, 40])

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, [ln], ncol=1)

    paper_pdf = PAPER_FIGS / "within_template_patching" / f"cot_boundary_{scope}_summary.pdf"
    results_png = layer_dir.parent / "summary_paper.png"
    _save(fig, paper_pdf, results_png, tight=False)


def _load_direct_scope(scope_dir: Path):
    """direct/{scope}/source_to_target.jsonl has all-layer effects per row."""
    path = scope_dir / "source_to_target.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    # Filter for absden_gt_0p5: |src_metric - tgt_metric| >= 0.5
    rows = [r for r in rows
            if abs(float(r.get("src_metric", 0)) - float(r.get("tgt_metric", 0))) >= 0.5]
    if not rows:
        return None
    effects = np.array([[e[0] for e in r["effects"]] for r in rows], dtype=float)
    cids = np.array([r["original_id"] for r in rows])
    _, tmpl = _template_means(effects, cids)
    lo, hi = _bootstrap_ci(tmpl)
    layers = np.arange(1, effects.shape[1] + 1)
    return dict(layers=layers, mean=tmpl.mean(axis=0), lo=lo, hi=hi,
                num_layers=effects.shape[1])


def render_within_template_patching_decomp():
    """Decomposition (direct, scope_compare): layer / mlp / attn_output."""
    base = (RESULTS / "gsm_sym_within_template_patching" /
            "llama-3.3-70b-instruct" / "direct")
    scopes = [
        ("layer",       "residual",  "#333333"),
        ("attn_output", "attention", "#1b7837"),
        ("mlp",         "MLP",       "#762a83"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    ax.axvspan(36, 40, color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    handles = []
    num_layers = None
    for scope, label, color in scopes:
        data = _load_direct_scope(base / scope)
        if data is None:
            continue
        num_layers = data["num_layers"]
        ax.fill_between(data["layers"], data["lo"], data["hi"],
                        color=color, alpha=0.18, linewidth=0)
        ln, = ax.plot(data["layers"], data["mean"], color=color, lw=1.3,
                      label=label)
        handles.append(ln)
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, num_layers or 80, [22, 36, 40])

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, handles, ncol=len(handles))

    paper_pdf = PAPER_FIGS / "within_template_patching" / "direct_scope_compare.pdf"
    results_png = base / "scope_compare_paper.png"
    _save(fig, paper_pdf, results_png, tight=False)


def render_within_template_patching_robust(model_short: str, stage_lines: list[int],
                                           scope: str = "value_tokens_position_exact"):
    """Fig 4a counterpart on robustness models.

    Reads cot_boundary/{scope}/layer/source_to_target_l*.jsonl (per-layer
    jsonls) and plots the residual patching curve with model-specific stage
    boundaries. stage_lines: [L_form, L_bind].
    """
    layer_dir = (RESULTS / "gsm_sym_within_template_patching" / model_short /
                 "cot_boundary" / scope / "layer")
    data = _load_per_layer_effects(layer_dir)
    if data is None:
        print(f"  [{model_short}] within-tmpl cot_boundary skipped: no data")
        return
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    sorted_lines = sorted(int(b) for b in stage_lines)
    if len(sorted_lines) >= 2:
        ax.axvspan(sorted_lines[0], sorted_lines[-1],
                   color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    color = "#333333"
    ax.fill_between(data["layers"], data["lo"], data["hi"],
                    color=color, alpha=0.20, linewidth=0)
    ln, = ax.plot(data["layers"], data["mean"], color=color, lw=1.3,
                  label="residual")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    _robust_stage_decorate(ax, stage_lines, data["num_layers"])
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _place_legend(ax, [ln], ncol=1)
    paper_pdf = (PAPER_FIGS / "within_template_patching" / "robust" /
                 model_short / f"cot_boundary_{scope}_summary.pdf")
    results_png = layer_dir.parent / "summary_robust.png"
    _save(fig, paper_pdf, results_png, tight=False)


def render_within_template_patching_decomp_robust(model_short: str,
                                                  stage_lines: list[int]):
    """Fig 4b counterpart on robustness models. Direct mode prompt_end
    decomposition (layer / attn_output / mlp). stage_lines: [L_form, L_bind].
    """
    base = (RESULTS / "gsm_sym_within_template_patching" / model_short / "direct")
    scopes = [
        ("layer",       "residual",  "#333333"),
        ("attn_output", "attention", "#1b7837"),
        ("mlp",         "MLP",       "#762a83"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    sorted_lines = sorted(int(b) for b in stage_lines)
    if len(sorted_lines) >= 2:
        ax.axvspan(sorted_lines[0], sorted_lines[-1],
                   color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    handles = []
    num_layers = None
    for scope, label, color in scopes:
        data = _load_direct_scope(base / scope)
        if data is None:
            print(f"  [{model_short}] decomp scope={scope}: missing")
            continue
        num_layers = data["num_layers"]
        ax.fill_between(data["layers"], data["lo"], data["hi"],
                        color=color, alpha=0.18, linewidth=0)
        ln, = ax.plot(data["layers"], data["mean"], color=color, lw=1.3,
                      label=label)
        handles.append(ln)
    if not handles:
        print(f"  [{model_short}] decomp skipped: no scopes")
        plt.close(fig)
        return
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    _robust_stage_decorate(ax, stage_lines, num_layers)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _place_legend(ax, handles, ncol=len(handles))
    paper_pdf = (PAPER_FIGS / "within_template_patching" / "robust" /
                 model_short / "direct_scope_compare.pdf")
    results_png = base / "scope_compare_robust.png"
    _save(fig, paper_pdf, results_png, tight=False)


# ---------------------------------------------------------------------------
# Family 6: within_template_logit_lens (Fig 7 Stage-4)
# ---------------------------------------------------------------------------

def render_within_template_logit_lens(mode: str):
    """mode: 'cot' or 'direct' — separate panels in main.tex."""
    base = RESULTS / "within_template_logit_lens" / "llama-3.3-70b-instruct"
    if mode == "cot":
        path = base / "within_template_logit_lens.jsonl"
        dst_name = "cot.pdf"
        png_name = "within_template_logit_lens_paper.png"
    else:
        path = base / "within_template_logit_lens_direct.jsonl"
        dst_name = "direct.pdf"
        png_name = "within_template_logit_lens_direct_paper.png"

    rows = [json.loads(l) for l in open(path) if l.strip()]
    own_per_pair, other_per_pair, cids = [], [], []
    for r in rows:
        lens_c = np.array(r["lens_c"], dtype=float)  # (L, 2)
        lens_w = np.array(r["lens_w"], dtype=float)
        own_per_pair.append((lens_c[:, 0] + lens_w[:, 0]) / 2)
        other_per_pair.append((lens_c[:, 1] + lens_w[:, 1]) / 2)
        cids.append(int(r["original_id"]))
    own_arr = np.stack(own_per_pair)
    other_arr = np.stack(other_per_pair)
    cids = np.array(cids)
    num_layers = own_arr.shape[1]
    layers = np.arange(1, num_layers + 1)

    _, own_tmpl = _template_means(own_arr, cids)
    _, oth_tmpl = _template_means(other_arr, cids)
    own_lo, own_hi = _bootstrap_ci(own_tmpl)
    oth_lo, oth_hi = _bootstrap_ci(oth_tmpl)

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.1))
    own_color = "#2166ac"
    oth_color = "#d6604d"
    ax.fill_between(layers, own_lo, own_hi, color=own_color, alpha=0.25, linewidth=0)
    ax.fill_between(layers, oth_lo, oth_hi, color=oth_color, alpha=0.25, linewidth=0)
    ln_o, = ax.plot(layers, own_tmpl.mean(axis=0), color=own_color, lw=1.3,
                    label="Own answer")
    ln_x, = ax.plot(layers, oth_tmpl.mean(axis=0), color=oth_color, lw=1.3,
                    label="Other answer")

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, num_layers, [22, 36, 40])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean log P(answer)")
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, [ln_o, ln_x], ncol=2)

    paper_pdf = PAPER_FIGS / "within_template_logit_lens" / dst_name
    results_png = base / png_name
    _save(fig, paper_pdf, results_png)


# ---------------------------------------------------------------------------
# Family 7: NoOp direct-mode supplementary panels (Stage 2 robustness)
# ---------------------------------------------------------------------------

def _load_noop_direct_curve(scope_dir: Path, direction_filename: str,
                            min_abs_denom: float = 0.5,
                            min_pairs_at_layer: int = 20):
    """Load + aggregate all-layer effects for one direction.

    Each row in `{direction_filename}.jsonl` has `effects: list[L]` where
    each entry is a 1-element list (per-row normalized patching effect at
    the patched layer; NaN at unrun layers). Per-pair effects are filtered
    by `abs(src_metric - tgt_metric) > min_abs_denom`. Aggregation at each
    layer is a NaN-safe pair-mean (skips NaN entries) — smoother than the
    template-mean-of-means when per-template counts are small. Bootstrap CI
    resamples PAIRS at each layer independently.

    Layers with fewer than `min_pairs_at_layer` non-NaN pairs are masked
    to NaN so the curve doesn't show spikes from 0-2 pair "samples".
    """
    path = scope_dir / f"{direction_filename}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows
            if abs(float(r.get("src_metric", 0.0))
                   - float(r.get("tgt_metric", 0.0))) > min_abs_denom]
    if not rows:
        return None
    effects = np.array([[e[0] for e in r["effects"]] for r in rows], dtype=float)
    n_pairs_per_layer = np.isfinite(effects).sum(axis=0)
    mean = np.nanmean(effects, axis=0)
    # Bootstrap CI per-layer over PAIRS (resample column independently).
    rng = np.random.default_rng(0)
    n_boot = 1000
    boot_means = np.empty((n_boot, effects.shape[1]), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, effects.shape[0], size=effects.shape[0])
        boot_means[b] = np.nanmean(effects[idx], axis=0)
    lo = np.nanquantile(boot_means, 0.025, axis=0)
    hi = np.nanquantile(boot_means, 0.975, axis=0)
    # Mask layers with too few pairs.
    insufficient = n_pairs_per_layer < min_pairs_at_layer
    mean[insufficient] = np.nan
    lo[insufficient] = np.nan
    hi[insufficient] = np.nan
    layers = np.arange(1, effects.shape[1] + 1)
    return dict(layers=layers, mean=mean, lo=lo, hi=hi,
                num_layers=effects.shape[1],
                n_pairs_per_layer=n_pairs_per_layer,
                n_rows=len(rows))


def render_noop_direct_scope_compare(pair_dataset: str, variant_tag: str):
    """Direct-mode prompt-end patching, sublayer (full / attn / MLP) traces.

    Shows the L40 attention-output spike (mover-head signature) in the
    NoOp-vs-filler patching trace, restore direction (wrong→correct).
    Aspect ratio matches the cot_swap overlay panels (SINGLE_COL_W × 1.6)
    so the digit-bearing Filler version sits alongside the cot_swap overlay
    panel in main Fig.~\\ref{fig:noop-stage2} at consistent height.
    """
    base = (RESULTS / "noop_patching" / "llama-3.3-70b-instruct" /
            pair_dataset / "direct")
    scopes = [
        ("layer",       "residual",  "#333333"),
        ("attn_output", "attention", "#1b7837"),
        ("mlp",         "MLP",       "#762a83"),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H),
                           constrained_layout=True)
    # Shade the L22-L36 Operation Planning band (same as the NoOp cliff
    # locus in panel a). Mirrors the L36-L40 shading style in
    # render_within_template_patching_decomp (Fig 4b).
    ax.axvspan(22, 36, color=CROSS_ONLY_COLOR, alpha=0.06, lw=0, zorder=0)
    handles = []
    num_layers = None
    for scope, label, color in scopes:
        data = _load_noop_direct_curve(base / scope, "wrong_to_correct")
        if data is None:
            print(f"  skipped: missing {pair_dataset}/{scope}")
            continue
        num_layers = data["num_layers"]
        ax.fill_between(data["layers"], data["lo"], data["hi"],
                        color=color, alpha=0.18, linewidth=0)
        ln, = ax.plot(data["layers"], data["mean"],
                      color=color, lw=1.3, label=label)
        handles.append(ln)
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, num_layers or 80, [22, 36, 40])

    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Patching effect", fontsize=7)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.set_ylim(-0.05, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, handles, ncol=len(handles))

    paper_pdf = (PAPER_FIGS / "noop_patching" /
                 f"direct_prompt_end_{variant_tag}.pdf")
    results_png = base / "prompt_end_scope_compare_paper.png"
    _save(fig, paper_pdf, results_png)


def render_noop_direct_qs_overlay(pair_dataset: str, variant_tag: str):
    """Direct-mode question-span patching, restore/disrupt overlay.

    Same scope as the canonical Stage 2 CoT figure (question span) but in
    direct mode. The L36→L40 second step in the digit-free variant is the
    Stage 3 echo discussed in the appendix text.
    """
    base = (RESULTS / "noop_patching" / "llama-3.3-70b-instruct" /
            pair_dataset / "direct" / "question_span" / "layer")
    panels = [
        ("wrong_to_correct", "Restore: filler$\\to$NoOp", "#1b7837"),
        ("correct_to_wrong", "Disrupt: NoOp$\\to$filler", "#762a83"),
    ]
    fig, ax = plt.subplots(figsize=(HALF_DOUBLE_W, 2.1))
    handles = []
    num_layers = None
    for direction, label, color in panels:
        data = _load_noop_direct_curve(base, direction)
        if data is None:
            print(f"  skipped: missing {pair_dataset}/qs/{direction}")
            continue
        num_layers = data["num_layers"]
        ax.fill_between(data["layers"], data["lo"], data["hi"],
                        color=color, alpha=0.22, linewidth=0)
        ln, = ax.plot(data["layers"], data["mean"],
                      color=color, lw=1.3, label=label)
        handles.append(ln)
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, num_layers or 80, [22, 36, 40])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized patching effect")
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, handles, ncol=len(handles))

    paper_pdf = (PAPER_FIGS / "noop_patching" /
                 f"direct_qs_overlay_{variant_tag}.pdf")
    results_png = base.parent / "qs_overlay_paper.png"
    _save(fig, paper_pdf, results_png)


def _load_noop_cot_boundary_curve(layer_dir: Path, direction: str,
                                   min_abs_denom: float = 0.5,
                                   clean_only: bool = False):
    """Load cot_boundary per-layer effects + free-gen recovery for one direction.

    Each row has `normalized_gold_logprob_effects: list[L][1]` (cliff metric)
    and `patched_final_answer_success: list[L][1]` (binary: did the patched-target
    freely emit gold at layer L?). Filter rows by |source_target_gap| > threshold.

    `clean_only`: drop pairs where patching-time labels disagree with the
    tag-time correct/wrong split (i.e. require source-correct + target-wrong
    for restore, or source-wrong + target-correct for disrupt). This makes
    `emit_gold` interpretable rather than confounded by per-pair label flips
    between batched-padded tag inference and unbatched patching inference.
    """
    path = layer_dir / f"{direction}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows
            if abs(float(r.get("source_target_gap", 0.0))) > min_abs_denom]
    if clean_only:
        # noop pipeline naming: "X_to_Y" means src=X, tgt=Y.
        if direction == "correct_to_wrong":
            # restore: src=filler_correct, tgt=noop_wrong → require src True, tgt False
            rows = [r for r in rows
                    if bool(r.get("unpatched_source_correct"))
                    and not bool(r.get("unpatched_target_correct"))]
        elif direction == "wrong_to_correct":
            # disrupt: src=noop_wrong, tgt=filler_correct → require src False, tgt True
            rows = [r for r in rows
                    if not bool(r.get("unpatched_source_correct"))
                    and bool(r.get("unpatched_target_correct"))]
    if not rows:
        return None
    eff_field = ("normalized_gold_logprob_effects"
                 if "normalized_gold_logprob_effects" in rows[0]
                 else "normalized_effects")

    def _coerce(entry):
        """Each per-layer entry is `[value]` or `None` (skipped layer)."""
        if entry is None:
            return np.nan
        if isinstance(entry, list):
            if not entry or entry[0] is None:
                return np.nan
            return float(entry[0])
        return float(entry)

    eff = np.array([[_coerce(e) for e in r[eff_field]] for r in rows], dtype=float)
    rec = np.array([[_coerce(e) for e in r["patched_final_answer_success"]]
                    for r in rows], dtype=float)
    cids = np.array([int(r["original_id"]) for r in rows])
    _, eff_tmpl = _template_means(eff, cids)
    _, rec_tmpl = _template_means(rec, cids)
    eff_lo, eff_hi = _bootstrap_ci(eff_tmpl)
    rec_lo, rec_hi = _bootstrap_ci(rec_tmpl)
    eff_mean = np.nanmean(eff_tmpl, axis=0)
    rec_mean = np.nanmean(rec_tmpl, axis=0)
    # Mask layers where coverage < 50% of pairs (otherwise mean of ~10% rows
    # is just noise and produces visual wobble at early layers). Coverage is
    # measured at the pair level, not the template level, so a layer is good
    # iff at least half the pairs have a valid value there.
    coverage = np.sum(~np.isnan(eff), axis=0) / max(1, eff.shape[0])
    sparse = coverage < 0.5
    for arr in (eff_mean, eff_lo, eff_hi, rec_mean, rec_lo, rec_hi):
        arr[sparse] = np.nan
    return dict(
        layers=np.arange(1, eff.shape[1] + 1), num_layers=eff.shape[1],
        eff_mean=eff_mean, eff_lo=eff_lo, eff_hi=eff_hi,
        rec_mean=rec_mean, rec_lo=rec_lo, rec_hi=rec_hi,
        coverage=coverage,
        n_rows=len(rows), n_templates=eff_tmpl.shape[0],
    )


def render_noop_freegen_two_panel(model_short: str = "llama-3.3-70b-instruct",
                                   pair_dataset: str = "filler_df_vs_noop_clean_tfm",
                                   variant_tag: str = "filler_df",
                                   stage_lines=(22, 36, 40),
                                   clean_only: bool = True):
    """Free-gen NoOp diagnosis: two panels overlaying restore + disrupt.

    (a) Normalized patching effect on source-answer logprob (Fig 5 cliff).
    (b) Free-gen recovery rate: fraction of pairs whose patched freely-generated
        answer matches source's gold.
    Both panels read from the noop_activation_patching cot_boundary pipeline.
    """
    base = (RESULTS / "noop_patching" / model_short / pair_dataset /
            "cot_boundary" / "question_span" / "layer")
    # Direction naming convention in the noop pipeline: "X_to_Y" means
    # source=X-side, target=Y-side. So `wrong_to_correct` is NoOp-wrong→filler-correct
    # (disrupt), and `correct_to_wrong` is filler-correct→NoOp-wrong (restore).
    panels = [
        ("correct_to_wrong", "Restore: filler$\\to$NoOp", "#1b7837"),
        ("wrong_to_correct", "Disrupt: NoOp$\\to$filler", "#762a83"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_W, 2.4))
    handles = []
    num_layers = None
    for direction, label, color in panels:
        data = _load_noop_cot_boundary_curve(base, direction, clean_only=clean_only)
        if data is None:
            print(f"  skipped: missing {pair_dataset}/cot_boundary/{direction}")
            continue
        num_layers = data["num_layers"]
        print(f"  {direction}: n_rows={data['n_rows']}, n_templates={data['n_templates']}")
        # Panel (b) shows "patching success rate":
        # - restore (correct_to_wrong, tgt=NoOp-wrong): emit_gold directly
        #   (restoring NoOp toward filler's correct emission)
        # - disrupt (wrong_to_correct, tgt=filler-correct): 1 − emit_gold
        #   (disrupting filler away from gold; HIGH = successfully disrupted)
        # Both curves then go HIGH (patch effective) → LOW (ineffective) across
        # the cliff, parallel to panel (a).
        flip_b = (direction == "wrong_to_correct")
        for ax_idx, (m_key, lo_key, hi_key) in enumerate(
            [("eff_mean", "eff_lo", "eff_hi"),
             ("rec_mean", "rec_lo", "rec_hi")]
        ):
            mean_vals = data[m_key].copy()
            lo_vals   = data[lo_key].copy()
            hi_vals   = data[hi_key].copy()
            if ax_idx == 1 and flip_b:
                # 1 - x for mean; CI bounds flip (1 - hi becomes new lo, 1 - lo becomes new hi)
                new_lo = 1.0 - hi_vals
                new_hi = 1.0 - lo_vals
                mean_vals = 1.0 - mean_vals
                lo_vals, hi_vals = new_lo, new_hi
            finite = np.isfinite(mean_vals)
            axes[ax_idx].fill_between(
                data["layers"], lo_vals, hi_vals,
                color=color, alpha=0.22, linewidth=0, where=finite,
            )
            ln, = axes[ax_idx].plot(
                data["layers"], mean_vals, color=color, lw=1.3, label=label,
            )
            if ax_idx == 0:
                handles.append(ln)

    panel_titles = [
        "(a) Patching effect", "(b) Free-gen patching success",
    ]
    panel_ylabels = [
        "Normalized effect on log P(src ans)",
        "Restore: emit gold;  disrupt: 1$-$emit gold",
    ]
    for ax, title, ylabel in zip(axes, panel_titles, panel_ylabels):
        ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
        ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")
        _draw_stage_lines(ax, stage_lines)
        _stage_xticks(ax, num_layers or 80, stage_lines)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=TICK_LS)
        ax.set_title(title, fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    _place_legend(axes[0], handles, ncol=len(handles))

    paper_pdf = (PAPER_FIGS / "noop_patching" /
                 f"freegen_two_panel_{model_short}_{variant_tag}.pdf")
    results_png = base.parent / "freegen_two_panel_paper.png"
    _save(fig, paper_pdf, results_png)


def render_noop_self_ww_patching():
    """NoOp wrong-wrong within-template cot_boundary cliff.

    Stage-3 specificity panel: pairs are both noop_clean-CoT-wrong, share
    template, differ in operand bindings. Gold per pair = source's
    regenerated wrong answer (set at baseline time). Mirrors the clean
    within-template Fig 4 cliff if Stage-3 binding is intact on
    noop-failure prompts.
    """
    layer_dir = (RESULTS / "noop_patching" / "llama-3.3-70b-instruct" /
                 "noop_clean_self_ww" / "cot_boundary" / "question_span" /
                 "layer")
    data = _load_per_layer_effects(layer_dir)
    if data is None:
        print("  skipped: no noop_clean_self_ww cot_boundary data")
        return

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, 2.1))
    color = "#333333"

    # Don't interpolate across missing layers — break the line at any gap
    # of more than one layer between consecutive samples.
    layers = data["layers"]
    mean   = data["mean"]
    lo     = data["lo"]
    his    = data["hi"]
    gap_starts = [0] + [i for i in range(1, len(layers)) if layers[i] - layers[i-1] > 1] + [len(layers)]
    for a, b in zip(gap_starts[:-1], gap_starts[1:]):
        seg_x = layers[a:b]
        if len(seg_x) == 0: continue
        ax.fill_between(seg_x, lo[a:b], his[a:b], color=color, alpha=0.20, linewidth=0)
        if len(seg_x) == 1:
            ax.plot(seg_x, mean[a:b], marker="o", ms=2.0, color=color, lw=0)
        else:
            ax.plot(seg_x, mean[a:b], color=color, lw=1.3)
    # Use a proxy handle for the legend so the segmented line shows one entry.
    from matplotlib.lines import Line2D
    ln = Line2D([0], [0], color=color, lw=1.3,
                label="Source→Target patching (NoOp✗↔NoOp✗)")
    ax.axhline(0, color="grey", lw=0.6, alpha=0.5, zorder=0)
    ax.axhline(1, color="grey", lw=0.6, alpha=0.4, zorder=0, ls=":")

    _draw_stage_lines(ax, [22, 36, 40])
    _stage_xticks(ax, data["num_layers"], [22, 36, 40])

    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized patching effect")
    ax.tick_params(axis="y", labelsize=TICK_LS)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _place_legend(ax, [ln], ncol=1)

    paper_pdf = (PAPER_FIGS / "noop_patching" /
                 "noop_clean_self_ww_cot_boundary_cliff.pdf")
    results_png = layer_dir.parent / "summary_paper.png"
    _save(fig, paper_pdf, results_png)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

FAMILIES = {
    "template_sim": lambda: [render_template_similarity(m) for m in ("direct", "cot")],
    "template_sim_robust": lambda: [
        # qwen-7B direct: 4 boundaries. S1=L19 (cross drop starts), S2=L21
        # (cross most-dropped while within still ~flat at 0.98), S3=L24 (joint
        # floor: within=0.78, cross=0.42, both bottomed before terminal bump),
        # S4=L28 (terminal). No plateau between S1 and S2 — cross drop is
        # monotonic.
        render_template_similarity_robust("qwen-math-7b-instruct", "direct",
            [(19, "cross_only"), (21, "cross_only"),
             (24, "within_cross"), (28, "within_cross")]),
        render_template_similarity_robust("qwen-math-7b-instruct", "cot",
            [(18, "cross_only"), (28, "within_cross")]),
        # gemma-9B: L_abs=11, L_form=27, L_bind=32 (number-token probe peak
        # plateau starts; jump at L29, peak L33; patching cliff at L30;
        # within-template cosine cliff at L31), L_out=42.
        render_template_similarity_robust("gemma-2-9b-it", "direct",
            [(11, "cross_only"), (27, "cross_only"),
             (32, "within_cross"), (42, "within_cross")]),
        render_template_similarity_robust("gemma-2-9b-it", "cot",
            [(27, "cross_only"), (42, "within_cross")]),
        # qwen-14B: L_abs=23, L_form=31, L_bind=37 (within-template cosine
        # cliff completes; patching cliff at L36; number-probe jump earlier
        # at L33 but the patching/cosine signals lag for qwen), L_out=48.
        render_template_similarity_robust("qwen-14b-instruct", "direct",
            [(23, "cross_only"), (31, "cross_only"),
             (37, "within_cross"), (48, "within_cross")]),
        render_template_similarity_robust("qwen-14b-instruct", "cot",
            [(37, "cross_only"), (48, "within_cross")]),
    ],
    "presence_probe_robust": lambda: [
        # Aligned with template_sim_robust: gemma L_bind=32 (probe peak
        # plateau), qwen L_bind=37 (cosine/patching cliff endpoint).
        render_presence_probe_robust("gemma-2-9b-it", [11, 27, 32, 42]),
        render_presence_probe_robust("qwen-14b-instruct", [23, 31, 37, 48]),
    ],
    "presence_probe": lambda: [
        # Headline (Fig 2): pairs with paired CV (matched-pair negatives).
        render_presence_probe("gsm_symbolic_pairs", "direct", "pairs"),
        render_presence_probe("gsm_symbolic_pairs", "cot",    "pairs"),
        # Appendix supplementary: pairs *data* but plain (unpaired) CV — gives
        # the unpaired probe ~6x more data per word than literal gsm_symbolic.
        # Skips panels for which the npz is not yet present (cot run in flight).
        # Fig 1 panel (b) is plain/direct and Fig 7 panel (b) is plain/cot:
        # both mark all four stage boundaries (L22 / L36 / L40 / L80) to align
        # with panel (a). legend_ncol=2 forces a 2-row legend so the panel's
        # aspect ratio matches panel (a) (template_similarity).
        # figsize_h: cot variant gets 2.0 so it matches template_similarity/cot
        # panel a's saved height (which renders taller due to wider y-range).
        # direct variant keeps 1.6 (matches Fig 2/Fig 1 panel a sizes).
        *[render_presence_probe("gsm_symbolic_pairs", m, "plain",
                                cv_path="template_disjoint_unpaired",
                                stage_lines=[22, 36, 40, 80],
                                legend_ncol=2,
                                figsize_h=(2.0 if m == "cot" else 1.6))
          for m in ("direct", "cot")
          if (RESULTS / "presence_probe" / "llama-3.3-70b-instruct" /
              "gsm_symbolic_pairs" / "template_disjoint_unpaired" / m /
              "correct" / "presence_probe.npz").exists()],
        # Fig 8 (presence-pre-reasoning): cot_pre_reasoning data lives under
        # the `all/` subset (no correctness-filtered variant). figsize_h is
        # bumped so the chart area is visually comparable to the Fig 7 pair
        # (Fig 8 is single-column while Fig 7 is two-column; matching the
        # per-panel visual height keeps the figure from looking diminutive
        # next to Fig 7).
        *([render_presence_probe("gsm_symbolic_pairs", "cot_pre_reasoning", "plain",
                                  cv_path="template_disjoint",
                                  stage_lines=[22, 36, 40, 80],
                                  legend_ncol=2,
                                  figsize_h=3.0,
                                  subset="all")]
          if (RESULTS / "presence_probe" / "llama-3.3-70b-instruct" /
              "gsm_symbolic_pairs" / "template_disjoint" / "cot_pre_reasoning" /
              "all" / "presence_probe.npz").exists() else []),
    ],
    "cot_swap_patching": lambda: [render_cot_swap_patching()],
    "p1_stage2_panel_a": lambda: [render_p1_stage2_panel_a(metric="logp")],
    "p1_stage2_panel_a_emit": lambda: [render_p1_stage2_panel_a(metric="emit")],
    "p1_stage2_panel_c": lambda: [render_p1_stage2_panel_c()],
    "stage2_robust": lambda: [
        render_stage2_robust("gemma-2-9b-it",    [11, 27]),
        render_stage2_robust("qwen-14b-instruct", [23, 31]),
    ],
    "noop_stage2_robust": lambda: [
        # Panel A: cot_boundary qs restore (single residual trace, Fig 4a style)
        render_noop_stage2_robust("gemma-2-9b-it",    [11, 27]),
        render_noop_stage2_robust("qwen-14b-instruct", [23, 31]),
        # Panel B: direct prompt_end decomposition (residual/attn/MLP)
        render_noop_direct_decomp_robust("gemma-2-9b-it",    [11, 27]),
        render_noop_direct_decomp_robust("qwen-14b-instruct", [23, 31]),
    ],
    "within_template_patching_robust": lambda: [
        # Aligned with template_sim/presence_probe: gemma [27, 32]
        # (probe peak plateau), qwen [31, 37] (cosine/patching cliff endpoint).
        render_within_template_patching_robust("gemma-2-9b-it",     [27, 32]),
        render_within_template_patching_robust("qwen-14b-instruct", [31, 37]),
        render_within_template_patching_decomp_robust("gemma-2-9b-it",     [27, 32]),
        render_within_template_patching_decomp_robust("qwen-14b-instruct", [31, 37]),
    ],
    "cot_swap_lens": lambda: [render_cot_swap_lens()],
    "within_template_patching": lambda: [
        render_within_template_patching("value_tokens_position_exact"),
        render_within_template_patching_decomp(),
    ],
    "within_template_logit_lens": lambda: [
        render_within_template_logit_lens("cot"),
        render_within_template_logit_lens("direct"),
    ],
    "noop_direct_supp": lambda: [
        render_noop_direct_scope_compare("filler_df_vs_noop_clean_tfm", "filler_df"),
        render_noop_direct_scope_compare("filler_vs_noop_tfm",          "filler"),
        render_noop_direct_qs_overlay("filler_df_vs_noop_clean_tfm",    "filler_df"),
        render_noop_direct_qs_overlay("filler_vs_noop_tfm",             "filler"),
    ],
    "noop_self_ww_patching": lambda: [render_noop_self_ww_patching()],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=list(FAMILIES) + ["all"], default="all")
    args = p.parse_args()
    setup_rc()
    targets = list(FAMILIES) if args.only == "all" else [args.only]
    for fam in targets:
        print(f"[{fam}]")
        FAMILIES[fam]()


if __name__ == "__main__":
    main()
