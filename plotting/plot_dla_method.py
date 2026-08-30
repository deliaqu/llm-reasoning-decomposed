"""Methodology figure for engagement-anchored DLA (Section 6).

Laid out as input, internals, output. The question is on top, the attention
heads in the middle, and the model's own CoT below. A head reads the distractor
clause C from the question and writes into the residual at the readout position
r in the CoT, and only that path is scored.

Two things have to be legible at once:
  * the anchor, which lives in the surface token stream. The target token t* and
    the readout position r are defined by where the CoT first engages the
    distractor, so the rows carry real text from a NoOp example.
  * the attribution, which lives inside the model, is per head and per layer,
    and is signed.

Box widths are measured from the rendered text, so the strings below can be
edited freely. Grid values are illustrative; real scores are in
results/dla_analysis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent


def _figdir():
    """Sibling paper tree when present, otherwise a figures/ dir in the repo."""
    p = REPO.parent / "paper" / "figures"
    return p if p.is_dir() else REPO / "figures"

C_CLS = "#2166ac"     # the NoOp clause
C_TGT = "#cc4125"     # readout position / the write we attribute
C_COT = "#b35806"     # the model's own CoT
C_GRID = "#999999"
C_POS = "#777777"
C_POS_H = "#c8443a"   # pro-engagement heads (H+), as in the per-layer figure
C_NEG_H = "#3b69b4"   # anti-engagement heads (H-)

FS_BOX, FS_LAB = 5.4, 5.6
XLIM, YLIM = (0.0, 10.0), (0.0, 3.3)
PADX, BOXH = 0.07, 0.34

VALS = [[0.1, -0.8, 0.0, 0.6, -0.2, 0.0, 0.3],
        [-0.5, 0.2, 0.9, -0.1, 0.0, -0.7, 0.1],
        [0.0, 0.7, -0.3, 0.2, -0.9, 0.4, 0.0]]


def _row(fig, ax, x0, y, items, gaps=None):
    """items: (text, facecolor|None, edgecolor, textcolor). Returns box spans.

    gaps[i] is the space left after box i, so tokens belonging to the same
    sentence can be drawn touching.
    """
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    gaps = gaps if gaps is not None else [0.10] * len(items)
    spans, x = [], x0
    for k, (txt, fc, ec, tc) in enumerate(items):
        t = ax.text(x + PADX, y + BOXH / 2, txt, fontsize=FS_BOX, style="italic",
                    ha="left", va="center", color=tc, zorder=4)
        bb = inv.transform_bbox(t.get_window_extent(renderer=r))
        w = bb.width + 2 * PADX
        ax.add_patch(Rectangle((x, y), w, BOXH, linewidth=0.7, edgecolor=ec,
                               facecolor=fc or "white", alpha=0.85 if fc else 1.0,
                               zorder=2))
        spans.append((x, x + w))
        x += w + gaps[k]
    return spans, x - gaps[-1]


def main():
    fig, ax = plt.subplots(figsize=(3.35, 1.82))
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_axis_off()
    fig.canvas.draw()

    X0 = 1.05

    # ---------- input: the NoOp question, distractor clause highlighted ----------
    y_q = 2.62
    q, _ = _row(fig, ax, X0, y_q, [
        ("Uma … 42-page report …", None, C_GRID, "black"),
        ("12 pages …", "#cfe0f0", C_CLS, C_CLS),
        ("How many …?", None, C_GRID, "black"),
    ])
    ax.annotate("", xy=(q[1][0], y_q + BOXH + 0.09), xytext=(q[1][1], y_q + BOXH + 0.09),
                arrowprops=dict(arrowstyle="-", color=C_CLS, lw=0.7))
    ax.text(sum(q[1]) / 2, y_q + BOXH + 0.14, r"NoOp clause $\mathcal{C}$",
            fontsize=FS_LAB, ha="center", va="bottom", color=C_CLS)
    ax.text(X0 - 0.12, y_q + BOXH / 2, "question", fontsize=FS_LAB, ha="right",
            va="center", color=C_POS)

    # ---------- output: the model's own CoT; t* opens the sentence using C ------
    y_c = 0.56
    # The period closing the previous sentence is the readout position r: the
    # last real token before t*. t* itself is only predicted there, so it is
    # drawn joined to the rest of the sentence it opens.
    c, _ = _row(fig, ax, X0, y_c, [
        ("She has written …", "#f2ddc4", C_COT, "black"),
        (".", "white", C_TGT, C_TGT),
        ("After", "#f6c3ba", C_TGT, C_TGT),
        ("subtracting the 12 …", "#fbe3df", C_TGT, "black"),
    ], gaps=[0.0, 0.12, 0.0, 0.0])
    ax.text(X0 - 0.12, y_c + BOXH / 2, "CoT", fontsize=FS_LAB, ha="right",
            va="center", color=C_POS)
    xr = sum(c[1]) / 2                        # centre of the r cell
    ax.text(xr, y_c - 0.05, r"$r$", fontsize=FS_LAB, ha="center", va="top",
            color=C_TGT)
    ax.text(sum(c[2]) / 2, y_c - 0.05, r"$t^{\star}$", fontsize=FS_LAB,
            ha="center", va="top", color=C_TGT)

    # ---------- internals: signed per-head, per-layer attribution ----------------
    nh, nl, cw, ch, g = len(VALS[0]), len(VALS), 0.34, 0.15, 0.05
    gw = nh * (cw + g) - g
    hx0, y_h = xr - 0.42 * gw, 1.44
    for l, row in enumerate(VALS):
        for i, v in enumerate(row):
            fc = C_POS_H if v > 0 else (C_NEG_H if v < 0 else "white")
            ax.add_patch(Rectangle((hx0 + i * (cw + g), y_h + l * (ch + g)), cw, ch,
                                   linewidth=0.6, edgecolor="#b0b0b0", facecolor=fc,
                                   alpha=min(abs(v) + 0.12, 1.0) if v else 1.0,
                                   zorder=3))
    gtop, gright = y_h + nl * (ch + g) - g, hx0 + gw
    ax.text(hx0 - 0.08, (y_h + gtop) / 2, "layer", fontsize=FS_LAB, ha="right",
            va="center", color=C_POS, rotation=90)
    ax.text(hx0, gtop + 0.03, "head", fontsize=FS_LAB, ha="left",
            va="bottom", color=C_POS)

    # up: the head's attention to the clause. down: the write it makes at r.
    ax.add_patch(FancyArrowPatch((xr, gtop + 0.05), (sum(q[1]) / 2, y_q - 0.05),
                                 connectionstyle="arc3,rad=-0.16", arrowstyle="-|>",
                                 mutation_scale=6.5, color=C_CLS, linewidth=0.9,
                                 ls="--", zorder=4))
    ax.text(xr - 0.14, (y_q + gtop) / 2 - 0.02, r"attends to $\mathcal{C}$",
            fontsize=FS_LAB, ha="right", va="center", color=C_CLS)
    ax.add_patch(FancyArrowPatch((xr, y_h - 0.05), (xr, y_c + BOXH + 0.04),
                                 arrowstyle="-|>", mutation_scale=6.5, color=C_TGT,
                                 linewidth=0.9, zorder=4))
    ax.text(xr + 0.14, (y_h + y_c + BOXH) / 2,
            "contributes to $\\log P(t^{\\star})$\nthrough that attention",
            fontsize=FS_LAB, ha="left", va="center", color=C_TGT, linespacing=1.4)

    # legend: what the sign of a head's score means. Sits under the grid so it
    # costs height rather than width, which is what the column budget cares about.
    sw, lx, ly = 0.18, X0, 0.10
    for col, lab in [(C_POS_H, r"$\mathcal{H}^{+}$ promotes $t^{\star}$"),
                     (C_NEG_H, r"$\mathcal{H}^{-}$ suppresses it")]:
        ax.add_patch(Rectangle((lx, ly - sw / 2), sw, sw, linewidth=0.6,
                               edgecolor="#b0b0b0", facecolor=col, zorder=3))
        t = ax.text(lx + sw + 0.07, ly, lab, fontsize=FS_LAB, ha="left",
                    va="center", color="#333333")
        fig.canvas.draw()
        lx = ax.transData.inverted().transform_bbox(
            t.get_window_extent(renderer=fig.canvas.get_renderer())).x1 + 0.28

    out_png = REPO / "results/dla_analysis/dla_method.png"
    out_pdf = _figdir() / "attention_mechanism/dla_method.pdf"
    for p in (out_png, out_pdf):
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=220, bbox_inches="tight", pad_inches=0.01)
        print(f"-> {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
