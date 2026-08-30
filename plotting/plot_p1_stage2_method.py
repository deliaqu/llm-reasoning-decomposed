"""Methodology figure for the Stage-2 (Operation Planning) patching experiments.

Deliberately near-wordless: the picture carries the structure (two runs, which
residual states are copied, where the effect is read) and the LaTeX caption
carries the explanation.

  (a) copy every question-span state at layer l; the model then generates, and
      rho is read at the marker it emits.
  (b) copy only the final injected-CoT state at layer l; rho is read at that
      same position, on the marker as next token.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

REPO = Path(__file__).resolve().parent.parent


def _figdir():
    """Sibling paper tree when present, otherwise a figures/ dir in the repo."""
    p = REPO.parent / "paper" / "figures"
    return p if p.is_dir() else REPO / "figures"

C_SRC, C_TGT, C_COT = "#b35806", "#2166ac", "#542788"
C_HOT, C_GRID = "#cc4125", "#c2c2c2"
CW, CH, G = 0.30, 0.30, 0.05
NL, L_PATCH = 3, 1


def _grid(ax, x0, y0, n, hi, hi_face):
    for c in range(n):
        x = x0 + c * (CW + G)
        for r in range(NL):
            on = (c in hi) and (r == L_PATCH)
            ax.add_patch(Rectangle((x, y0 + r * (CH + G)), CW, CH, linewidth=0.7,
                                   edgecolor=C_HOT if on else C_GRID,
                                   facecolor=hi_face if on else "white",
                                   alpha=0.85 if on else 1.0, zorder=2))
    w = n * (CW + G) - G
    return [x0 + c * (CW + G) + CW / 2 for c in range(n)], w


def _brace(ax, x0, x1, y, text, col, drop=0.05):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-", color=col, lw=0.7))
    ax.text((x0 + x1) / 2, y - drop, text, fontsize=5.7, ha="center",
            va="top", color=col)


def _panel(ax, nq, ncot, hi, gen):
    n = nq + ncot
    y0 = 0.62
    gw = n * (CW + G) - G
    x_src, x_tgt = 0.0, gw + 1.35
    right = x_tgt + gw + (1.95 if gen else 0.75)
    ax.set_xlim(-0.62, right); ax.set_ylim(y0 - 0.50, y0 + NL * (CH + G) + 0.13)
    ax.axis("off")


    # layer axis
    ax.annotate("", xy=(-0.20, y0 + NL * (CH + G) - G), xytext=(-0.20, y0),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=0.7))
    ax.text(-0.26, y0, "L0", fontsize=5.4, ha="right", va="bottom", color="#777")
    ax.text(-0.26, y0 + NL * (CH + G) - G, "L80", fontsize=5.4, ha="right",
            va="top", color="#777")
    ax.text(-0.26, y0 + L_PATCH * (CH + G) + CH / 2, r"$\ell$", fontsize=6.4,
            ha="right", va="center", color=C_HOT)

    xs_s, _ = _grid(ax, x_src, y0, n, hi, C_SRC)
    xs_t, _ = _grid(ax, x_tgt, y0, n, hi, C_SRC)

    for xs, qname, qcol in ((xs_s, "padded-Sym. question", C_SRC),
                            (xs_t, "P1 question", C_TGT)):
        _brace(ax, xs[0] - CW / 2, xs[nq - 1] + CW / 2, y0 - 0.10, qname, qcol)
        if ncot:
            _brace(ax, xs[nq] - CW / 2, xs[-1] + CW / 2, y0 - 0.10,
                   "padded-Sym. CoT", C_COT, drop=0.26)

    y_l = y0 + L_PATCH * (CH + G) + CH / 2
    ax.add_patch(FancyArrowPatch((x_src + gw + 0.10, y_l), (x_tgt - 0.10, y_l),
                                 arrowstyle="-|>", mutation_scale=9, color=C_HOT,
                                 linewidth=1.2, zorder=5))
    ax.text((x_src + gw + x_tgt) / 2, y_l + 0.07, "copy", fontsize=6.0,
            ha="center", va="bottom", color=C_HOT)

    y_top_cell = y0 + (NL - 1) * (CH + G)
    if gen:
        # the model's own continuation: further positions of the same sequence,
        # drawn dotted; the readout is the marker it emits
        ngen = 3
        gx0 = x_tgt + gw + G
        for c in range(ngen):
            gx = gx0 + c * (CW + G)
            ax.add_patch(Rectangle((gx, y0 + (NL - 1) * (CH + G)), CW, CH,
                                   linewidth=0.7, ls=(0, (1.5, 1.2)),
                                   edgecolor="#a8a8a8", facecolor="white",
                                   zorder=2))
        ax.text(gx0 + 0.10, y0 + (NL - 1) * (CH + G) - 0.10, "generated",
                fontsize=5.7, ha="left", va="top", color="#999")
        xr = gx0 + (ngen - 1) * (CW + G)
    else:
        xr = xs_t[-1] - CW / 2
    ax.add_patch(Rectangle((xr, y_top_cell), CW, CH, linewidth=1.1,
                           ls=(0, (2, 1.4)), edgecolor=C_HOT, facecolor="white",
                           zorder=4))
    lab = r"read $\rho$(answer)" if gen else r"read $\rho$(####)"
    ax.text(xr + CW / 2, y0 + NL * (CH + G) + 0.02, lab, fontsize=5.7,
            ha="center", va="bottom", color=C_HOT)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 0.80),
                             gridspec_kw={"width_ratios": [1.0, 1.10],
                                          "wspace": 0.05})
    _panel(axes[0], nq=8, ncot=0, hi=set(range(8)), gen=True)
    _panel(axes[1], nq=8, ncot=4, hi={11}, gen=False)
    fig.canvas.draw()
    bb0, bb1 = axes[0].get_position(), axes[1].get_position()
    sep = (bb0.x1 + bb1.x0) / 2
    fig.lines.append(plt.Line2D([sep, sep], [0.05, 0.95], transform=fig.transFigure,
                                color="#ccc", lw=0.6))
    png = REPO / "results/cot_swap_activation_patching/llama-3.3-70b-instruct/p1_stage2_methodology.png"
    pdf = _figdir() / "cot_swap_patching/p1_vs_padded_symbolic_methodology.pdf"
    for path in (png, pdf):
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.01)
        print(f"-> {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
