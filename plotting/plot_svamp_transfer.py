"""SVAMP cross-benchmark transfer figure (appendix companion to Fig. 2b).

Draws the four stage signatures on SVAMP against the layer axis, so the shapes
can be compared with the GSM-Symbolic panel rather than sampled at a few points.

The four curves do NOT come from one run, and the figure says so:
  * Entity and Operator come from SVAMP proper.
  * Number tokens come from svamp_variants, a re-instantiated number-bank
    version of the same problems.
  * Final answer and Number tokens are both within-problem contrasts on the
    variants: negatives are other variants of the same problem. What sets the
    final-answer curve apart, and why it is drawn with markers, is that it
    comes from a separate diagnostic evaluated at 13 sampled layers, not 81.

Values reproduce Table 9 exactly: 0.71->0.51, 0.90/0.98/0.98, 0.59->0.71,
0.58->0.95.
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent


def _figdir():
    """Sibling paper tree when present, otherwise a figures/ dir in the repo."""
    p = REPO.parent / "paper" / "figures"
    return p if p.is_dir() else REPO / "figures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_render import (setup_rc, SINGLE_COL_W, PANEL_H, TICK_LS,  # noqa: E402
                          PRESENCE_COLORS, PRESENCE_LABELS,
                          _presence_bootstrap_ci, _draw_stage_lines,
                          _stage_xticks, _place_legend)

PROBE = REPO / "results/presence_probe/Llama-3.3-70B-Instruct"
SVAMP = PROBE / "svamp/seed_disjoint/direct/correct/svamp_presence_probe.npz"
VARIANTS = PROBE / "svamp_variants/seed_disjoint/direct/correct/svamp_variants_presence_probe.npz"
ANSWER = PROBE / "svamp_variants/seed_disjoint/direct/correct/svamp_answer_within_problem_diag.json"
STAGES = [22, 36, 40, 80]


def _family(npz, prefix):
    """(layers, K) matrix of per-token probe curves for one family."""
    keys = [k for k in npz.files if k.startswith(prefix + "|")]
    return np.stack([npz[k] for k in keys], axis=1)


def main():
    setup_rc()
    sv, va = np.load(SVAMP, allow_pickle=True), np.load(VARIANTS, allow_pickle=True)
    curves = [("entity", _family(sv, "entity_decorative")),
              ("op_presence", _family(sv, "op_presence")),
              ("number", _family(va, "number_bank"))]

    fig, ax = plt.subplots(figsize=(SINGLE_COL_W, PANEL_H))
    handles = []
    for name, arr in curves:
        mean, lo, hi = _presence_bootstrap_ci(arr)
        layers = np.arange(arr.shape[0])
        col = PRESENCE_COLORS[name]
        ax.fill_between(layers, lo, hi, alpha=0.20, color=col, linewidth=0)
        ln, = ax.plot(layers, mean, color=col, lw=1.2, label=PRESENCE_LABELS[name])
        handles.append(ln)

    # Final answer: a different probe design on sampled layers, so markers.
    d = json.load(open(ANSWER))
    aL = np.array(d["layers"])
    a = np.array([[t[str(L)] for L in aL] for t in d["per_target"].values()])
    ln, = ax.plot(aL, a.mean(axis=0), color=PRESENCE_COLORS["answer"], lw=1.2,
                  ls="--", marker="o", ms=2.2, mew=0,
                  label=PRESENCE_LABELS["answer"] + " (13 layers)")
    handles.append(ln)

    ax.axhline(0.5, color="#888888", lw=0.8, ls=":", alpha=0.9)
    _draw_stage_lines(ax, STAGES)
    ax.set_xlim(0, 80)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Probe accuracy", fontsize=7)
    ax.set_yticks([0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=TICK_LS)
    _stage_xticks(ax, 80, STAGES)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _place_legend(ax, handles, ncol=2)

    out_pdf = _figdir() / "cross_dataset/svamp_transfer.pdf"
    out_png = REPO / "results/presence_probe/Llama-3.3-70B-Instruct/svamp_transfer.png"
    for p in (out_pdf, out_png):
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.01)
        print(f"-> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
