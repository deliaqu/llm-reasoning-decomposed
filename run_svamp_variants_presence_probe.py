"""Within-template number-presence probes on SVAMP operand variants.

The GSM-style surgical S3 instrument for SVAMP. The plain SVAMP number probe
is a cross-problem contrast, so number presence is entangled with problem
topic (12<->dozens). Here each problem appears as 4 operand variants
(data_scripts/svamp/build_svamp_variants.py), with distinctive 2-digit bank
values injected — so for a probed value w, positives and negatives include
THE SAME narrative with and without w (95-166 contrast templates per value
among direct-correct rows). This isolates operand presence from topic.

What it adjudicates (see the GSM-vs-SVAMP number-curve shape question):
  - if SVAMP's flattened L40 step was instrument-driven, it sharpens here;
  - if the late rise is depth-driven (1-2-op operands stay computation-
    relevant to emission), it persists even with the clean instrument.

Categories:
  number_bank      16 bank-value presence probes (the instrument)
  number_bank_am   same, answer-distribution-matched (note: operand value
                   causally moves the answer within a template, so matching
                   is conservative here — it removes real signal along with
                   any leak; report both)
  op_presence      formula operators (sanity anchor; unchanged per template)

CV: GroupKFold(seed_group) — variants inherit their problem's seed_group, so
all variants of a problem (and its variation cluster) stay in one fold.
Population: direct-correct rows. Fit machinery (10-fold, 5 reps, bootstrap
CI of category mean) imported from run_svamp_presence_probe.

Usage:
  python run_svamp_variants_presence_probe.py            # extract + fit + plot
  python run_svamp_variants_presence_probe.py --fit_only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "disentangled_evaluation"))
from paths import CACHE_DIR, DATA_DIR, RESULT_DIR  # noqa: E402

import run_svamp_presence_probe as SP  # noqa: E402  (fit machinery + extraction contract)

MODEL_NAME = SP.MODEL_NAME
MODEL_SHORT = SP.MODEL_SHORT

RESULTS_CSV = (
    f"{RESULT_DIR}/transformers_direct/svamp_variants_results_{MODEL_SHORT}_transformers.csv"
)
SRC_CSV = f"{DATA_DIR}/test_svamp_variants.csv"
HS_CACHE_DIR = Path(CACHE_DIR) / "svamp_variants_probe"
HS_CACHE = HS_CACHE_DIR / f"hidden_states_{MODEL_SHORT}_direct.npz"

OUT_DIR = (
    Path(RESULT_DIR).parent
    / "presence_probe" / MODEL_SHORT / "svamp_variants"
    / "seed_disjoint" / "direct" / "correct"
)

VALUE_BANK = [17, 23, 29, 34, 38, 41, 47, 53, 59, 62, 67, 71, 76, 83, 89, 94]
SEED = 0

CATEGORY_STYLE = {
    "number_bank": ("tab:green", "S3 numbers, within-template (K=16)"),
    "number_bank_am": ("seagreen", "same, answer-matched (conservative)"),
    "op_presence": ("tab:orange", "S2 formula operators (anchor)"),
}


def load_frames():
    res = pd.read_csv(RESULTS_CSV)
    src = pd.read_csv(SRC_CSV)
    assert len(res) == len(src) == 3814
    assert (res["question"] == src["question"]).all()
    correct = res["original_direct_correctness"].fillna(0).astype(bool).to_numpy()
    return res, correct


def build_targets(sub: pd.DataFrame):
    groups = sub["seed_group"].to_numpy()
    num_sets = [set(re.findall(r"\d+", q)) for q in sub["question"].astype(str)]

    bank_targets = {}
    for w in VALUE_BANK:
        y = np.array([str(w) in s for s in num_sets])
        # viability is guaranteed by the injection design; assert, don't gate
        assert y.sum() >= 50 and len(np.unique(groups[y])) >= 20, f"bank {w} too thin"
        bank_targets[f"num:{w}"] = y

    ops_rows = [re.findall(r"[+\-*/]", str(f)) for f in sub["formula"]]
    op_targets = {
        f"op:{op}": np.array([op in r for r in ops_rows])
        for op in ["+", "-", "*", "/"]
    }

    categories = {
        "number_bank": bank_targets,
        "number_bank_am": dict(bank_targets),
        "op_presence": op_targets,
    }
    return categories, groups


def plot(results: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rng = np.random.default_rng(SEED)
    for cat, targets in results.items():
        color, label = CATEGORY_STYLE[cat]
        curves = np.stack(list(targets.values()))
        mean = np.nanmean(curves, axis=0)
        x = np.arange(curves.shape[1])
        ax.plot(x, mean, color=color, label=label, lw=1.8)
        if curves.shape[0] > 1:
            K = curves.shape[0]
            boot = np.nanmean(curves[rng.integers(0, K, size=(2000, K))], axis=1)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    for layer, name in SP.STAGE_LINES.items():
        ax.axvline(layer, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.text(layer, 1.02, name, ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("balanced accuracy (seed-disjoint CV)")
    ax.set_title(f"SVAMP variants: within-template number presence — {MODEL_SHORT}\n"
                 "(shading: 95% bootstrap CI of category mean)", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fit_only", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    res, correct = load_frames()
    print(f"direct-correct: {correct.sum()}/3814")

    if args.fit_only:
        hs = np.load(HS_CACHE)["hidden_states"]
        print(f"loaded cached hidden states {hs.shape}")
    else:
        # numeric-answer dataset -> same 'original' direct instruction as
        # svamp inference; SP.extract_hidden_states mirrors that exactly.
        # Redirect its cache path to this dataset's cache.
        SP.HS_CACHE_DIR, SP.HS_CACHE = HS_CACHE_DIR, HS_CACHE
        hs = SP.extract_hidden_states(res["question"].astype(str).tolist(), args.batch_size)
    assert hs.shape[0] == 3814

    sub = res[correct].reset_index(drop=True)
    categories, groups = build_targets(sub)
    for cat, t in categories.items():
        print(f"  {cat:15s}: {len(t)} targets")

    ans = sub["answer"].astype(float).to_numpy()
    quantiles = np.quantile(ans, np.linspace(0, 1, SP.N_ANSWER_MATCH_BINS + 1)[1:-1])
    match_bin = np.searchsorted(quantiles, ans)
    results = SP.fit_all(hs[correct], categories, groups, args.n_jobs,
                         match_bin=match_bin, matched_categories={"number_bank_am"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "svamp_variants_presence_probe.npz",
        **{f"{cat}|{name}": acc for cat, t in results.items() for name, acc in t.items()},
    )
    meta_out = {
        "model": MODEL_NAME,
        "results_csv": RESULTS_CSV,
        "n_correct": int(correct.sum()),
        "cv": "GroupKFold(seed_group); variants inherit their problem's group",
        "value_bank": VALUE_BANK,
        "note": "answer-matched variant is conservative here: operand value "
                "causally moves the answer within a template",
    }
    (OUT_DIR / "svamp_variants_presence_probe_meta.json").write_text(
        json.dumps(meta_out, indent=2))
    plot(results, OUT_DIR / "svamp_variants_presence_probe.png")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
