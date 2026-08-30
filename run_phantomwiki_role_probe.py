"""Hypothesis-free word-role probes on PhantomWiki name variants.

No stage assignment a priori: every category is defined by a MECHANICAL
criterion (where the token occurs / what role the solution trace assigns it),
all categories are probed identically across all 81 layers, and the
trajectory shapes are the finding. Stage interpretation happens after, in
analysis — not in the design.

Name categories (per bank first-name, role from build_phantomwiki_role_variants):
  name_anchor        appears in the question (the chain's lookup key)
  name_intermediate  on the solution chain, not in question, not the answer
  name_answer        the answer entity (who-questions)
  name_bystander     in this copy's corpus, none of the above
  name_inserted      in the filler insertion adjacent to the question
Other categories:
  value_inserted     hobby/occupation value in the filler insertion
  relation           relation words in the question (mother, friend, ...)
  answer_count       gold count-answer value (count-answer rows only)

Target construction: for name w in role R, positives = rows where w fills
role R; negatives = rows where w is ABSENT from the prompt entirely; rows
where w is present in any other role are EXCLUDED (per-target row mask), so
each curve isolates one role. CV: GroupKFold(universe) — all 3 name-copies
of a universe share its group, so folds hold out whole graph families.

Population: direct-correct rows of test_phantomwiki_roles.csv.

Usage:
  python run_phantomwiki_role_probe.py                # extract + fit + plot
  python run_phantomwiki_role_probe.py --fit_only
  # data-parallel extraction (mirrors run_inference_transformers_direct):
  python run_phantomwiki_role_probe.py --num_shards 4 --shard_index 0
  python run_phantomwiki_role_probe.py --fit_only --num_shards 4  # merge + fit
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "disentangled_evaluation"))
from paths import CACHE_DIR, DATA_DIR, RESULT_DIR  # noqa: E402

import run_phantomwiki_presence_probe as PW  # noqa: E402  (extraction contract: original_text)
import run_svamp_presence_probe as SP  # noqa: E402  (fit machinery)

MODEL_NAME = SP.MODEL_NAME
MODEL_SHORT = SP.MODEL_SHORT

RESULTS_CSV = (
    f"{RESULT_DIR}/transformers_direct/phantomwiki_roles_results_{MODEL_SHORT}_transformers.csv"
)
SRC_CSV = f"{DATA_DIR}/test_phantomwiki_roles.csv"
HS_CACHE_DIR = Path(CACHE_DIR) / "phantomwiki_roles_probe"
HS_CACHE = HS_CACHE_DIR / f"hidden_states_{MODEL_SHORT}_direct.npz"


def shard_cache(shard_index: int, num_shards: int) -> Path:
    return HS_CACHE_DIR / (
        f"hidden_states_{MODEL_SHORT}_direct.shard{shard_index}of{num_shards}.npz"
    )

OUT_DIR = (
    Path(RESULT_DIR).parent
    / "presence_probe" / MODEL_SHORT / "phantomwiki_roles"
    / "universe_disjoint" / "direct" / "correct"
)

MIN_POS = 15
MIN_POS_UNIVERSES = 6
MAX_TARGETS = 20
SEED = 0

ROLE_COLUMNS = {
    "name_anchor": "role_anchor",
    "name_intermediate": "role_intermediate",
    "name_answer": "role_answer",
    "name_bystander": "role_bystander",
    "name_inserted": "inserted_names",
}

# descriptive styling only — no stage labels anywhere
CATEGORY_STYLE = {
    "name_anchor": ("tab:blue", "name: anchor (in question)"),
    "name_intermediate": ("tab:cyan", "name: chain intermediate"),
    "name_answer": ("tab:purple", "name: answer entity"),
    "name_bystander": ("tab:gray", "name: bystander (corpus only)"),
    "name_inserted": ("tab:brown", "name: inserted filler"),
    "value_inserted": ("tab:pink", "value: inserted filler"),
    "relation": ("tab:orange", "relation words (in question)"),
    "answer_count": ("tab:green", "count-answer value"),
}


def load_frames():
    res = pd.read_csv(RESULTS_CSV)
    src = pd.read_csv(SRC_CSV)
    assert len(res) == len(src), (len(res), len(src))
    assert (res["pw_id"] == src["pw_id"]).all()
    assert (res["question"] == src["question"]).all()
    correct = res["original_direct_correctness"].fillna(0).astype(bool).to_numpy()
    return res, correct


def build_targets(sub: pd.DataFrame):
    """Returns categories: cat -> {target -> (y, row_mask)} and groups."""
    groups = sub["universe"].to_numpy()
    n = len(sub)

    role_sets = {cat: [set(json.loads(s)) for s in sub[col]]
                 for cat, col in ROLE_COLUMNS.items()}
    # The builder picks inserted names FROM the row's bystanders and leaves
    # them in role_bystander; insertion supersedes bystander for that row.
    for i in range(n):
        role_sets["name_bystander"][i] -= role_sets["name_inserted"][i]
    present_any = [set().union(*(role_sets[c][i] for c in role_sets)) for i in range(n)]
    # Names filling >=2 roles in one row (two people sharing a first name,
    # e.g. one on the chain and one a bystander) are ambiguous there; the
    # design rule "rows where it fills a different role are excluded" makes
    # such rows ineligible as positives OR negatives for that name.
    multi_role = []
    for i in range(n):
        counts = defaultdict(int)
        for cset in role_sets.values():
            for w in cset[i]:
                counts[w] += 1
        multi_role.append({w for w, k in counts.items() if k > 1})

    name_pool = defaultdict(int)
    for cat in ("name_anchor", "name_answer", "name_intermediate", "name_inserted"):
        for s in role_sets[cat]:
            for w in s:
                name_pool[w] += 1

    def viable(y, mask):
        pos = y & mask
        if pos.sum() < MIN_POS or (mask & ~y).sum() < MIN_POS:
            return False
        return len(np.unique(groups[pos])) >= MIN_POS_UNIVERSES

    categories: dict[str, dict] = {}
    for cat in ROLE_COLUMNS:
        targets = {}
        for w in sorted(name_pool, key=lambda k: -name_pool[k]):
            y = np.array([w in role_sets[cat][i] for i in range(n)])
            # exclude rows where w is present in a DIFFERENT role, and rows
            # where w is role-ambiguous (fills the target role AND another)
            other = np.array([w in present_any[i] for i in range(n)]) & ~y
            ambig = np.array([w in multi_role[i] for i in range(n)])
            mask = ~(other | ambig)
            if viable(y, mask):
                targets[f"{cat[5:] if cat.startswith('name_') else cat}:{w}"] = (y, mask)
            if len(targets) >= MAX_TARGETS:
                break
        categories[cat] = targets

    # inserted attribute values (multi-word strings; presence = row inserted it)
    val_rows = [set(json.loads(s)) for s in sub["inserted_values"]]
    val_pool = defaultdict(int)
    for s in val_rows:
        for v in s:
            val_pool[v] += 1
    vtargets = {}
    for v in sorted(val_pool, key=lambda k: -val_pool[k]):
        y = np.array([v in val_rows[i] for i in range(n)])
        mask = np.ones(n, dtype=bool)
        if viable(y, mask):
            vtargets[f"val:{v[:24]}"] = (y, mask)
        if len(vtargets) >= MAX_TARGETS:
            break
    categories["value_inserted"] = vtargets

    rq = sub["raw_question"].astype(str)
    rel_targets = {}
    for r in PW.RELATIONS:
        y = rq.str.contains(rf"\b{r}s?\b", case=False, regex=True).to_numpy()
        mask = np.ones(n, dtype=bool)
        if viable(y, mask):
            rel_targets[f"rel:{r}"] = (y, mask)
    categories["relation"] = rel_targets

    is_count = sub["answer"].astype(str).str.fullmatch(r"\d+").to_numpy()
    ans = np.where(is_count, sub["answer"].astype(str), "")
    ans_targets = {}
    for v in ["1", "2", "3", "4", "5"]:
        y = (ans == v) & is_count
        if viable(y, is_count):
            ans_targets[f"ans:{v}"] = (y, is_count.copy())
    categories["answer_count"] = ans_targets
    return categories, groups


def _fit_layer_masked(X_layer, flat_targets, groups):
    out = []
    for (_, _, y, mask) in flat_targets:
        out.append(SP.fit_target_layer(X_layer[mask], y[mask], groups[mask], SP.SEED))
    return out


def fit_all_masked(hs, categories, groups, n_jobs):
    from joblib import Parallel, delayed
    n_layers = hs.shape[1]
    flat = [(cat, name, y, mask)
            for cat, targets in categories.items() for name, (y, mask) in targets.items()]
    print(f"fitting {len(flat) * n_layers} probes ({len(flat)} targets x {n_layers} layers)")
    per_layer = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_layer_masked)(hs[:, l, :].astype(np.float32), flat, groups)
        for l in range(n_layers)
    )
    results: dict[str, dict[str, list]] = defaultdict(dict)
    for l, accs in enumerate(per_layer):
        for (cat, name, _, _), acc in zip(flat, accs):
            results[cat].setdefault(name, [np.nan] * n_layers)[l] = acc
    return {c: {n_: np.array(v) for n_, v in t.items()} for c, t in results.items()}


def plot(results, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5))
    rng = np.random.default_rng(SEED)
    for cat, targets in results.items():
        if not targets:
            continue
        color, label = CATEGORY_STYLE[cat]
        curves = np.stack(list(targets.values()))
        mean = np.nanmean(curves, axis=0)
        x = np.arange(curves.shape[1])
        ax.plot(x, mean, color=color, label=f"{label} (K={len(targets)})", lw=1.8)
        if curves.shape[0] > 1:
            K = curves.shape[0]
            boot = np.nanmean(curves[rng.integers(0, K, size=(2000, K))], axis=1)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0)
    for layer in (22, 36, 40, 80):
        ax.axvline(layer, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.text(layer, 1.02, f"L{layer}", ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("balanced accuracy (universe-disjoint CV)")
    ax.set_title(f"PhantomWiki word-role trajectories — {MODEL_SHORT}, direct, correct-only\n"
                 "(categories defined mechanically; no stage assignment a priori)", fontsize=10)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--fit_only", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    # Data-parallel extraction sharding (mirrors run_inference_transformers_direct):
    # each shard job forward-passes a contiguous row slice and writes a
    # shard-suffixed npz; `--fit_only --num_shards N` concatenates the shards
    # (shard order == row order) and fits. Defaults (1, 0) => unsharded.
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    args = ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        sys.exit(f"Invalid shard config: shard_index={args.shard_index}, "
                 f"num_shards={args.num_shards}")

    res, correct = load_frames()
    print(f"direct-correct: {correct.sum()}/{len(res)}")

    # ONE-PASS DESIGN: prefer the residual cache written by inference itself
    # (run_inference_transformers_direct --save-by-default). Probe-side
    # extraction is a legacy fallback for datasets whose inference predates
    # the one-pass capture (e.g. this one, inferenced 2026-07-10 pre-fix).
    from run_inference_transformers_direct import hidden_states_path
    onepass = hidden_states_path("phantomwiki_roles", f"{MODEL_SHORT}_transformers")

    if args.num_shards > 1 and not args.fit_only:
        if Path(onepass).exists():
            sys.exit("one-pass cache exists — sharded extraction is unnecessary; "
                     "run without --num_shards.")
        n = len(res)
        lo = (n * args.shard_index) // args.num_shards
        hi = (n * (args.shard_index + 1)) // args.num_shards
        out = shard_cache(args.shard_index, args.num_shards)
        print(f"extraction shard {args.shard_index}/{args.num_shards}: "
              f"rows [{lo}:{hi}) -> {out}")
        PW.HS_CACHE_DIR, PW.HS_CACHE = HS_CACHE_DIR, out
        hs = PW.extract_hidden_states(
            res["question"].astype(str).tolist()[lo:hi], args.batch_size)
        assert hs.shape[0] == hi - lo
        return

    if Path(onepass).exists():
        hs = np.load(onepass)["hidden_states"]
        print(f"loaded ONE-PASS inference cache {hs.shape} from {onepass}")
    elif args.fit_only:
        if args.num_shards > 1 and not HS_CACHE.exists():
            parts = [np.load(shard_cache(i, args.num_shards))["hidden_states"]
                     for i in range(args.num_shards)]
            hs = np.concatenate(parts, axis=0)
            print(f"assembled {args.num_shards} extraction shards -> {hs.shape}")
            np.savez(HS_CACHE, hidden_states=hs)  # uncompressed: fast, refit-friendly
            print(f"wrote merged cache -> {HS_CACHE}")
        else:
            hs = np.load(HS_CACHE)["hidden_states"]
    else:
        print("WARNING: no one-pass cache found — falling back to probe-side "
              "extraction (legacy; only valid for pre-fix inference runs).")
        PW.HS_CACHE_DIR, PW.HS_CACHE = HS_CACHE_DIR, HS_CACHE
        hs = PW.extract_hidden_states(res["question"].astype(str).tolist(), args.batch_size)
    assert hs.shape[0] == len(res)

    sub = res[correct].reset_index(drop=True)
    categories, groups = build_targets(sub)
    for cat, t in categories.items():
        print(f"  {cat:18s}: {len(t):2d} targets | {sorted(t)[:5]}")

    results = fit_all_masked(hs[correct], categories, groups, args.n_jobs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "phantomwiki_role_probe.npz",
             **{f"{c}|{n_}": a for c, t in results.items() for n_, a in t.items()})
    meta = {
        "model": MODEL_NAME,
        "results_csv": RESULTS_CSV,
        "n_fit": int(correct.sum()),
        "cv": "GroupKFold(universe); name-copies share their universe's group",
        "design": "hypothesis-free: categories mechanical, stage reading post-hoc",
        "gate": {"min_pos": MIN_POS, "min_pos_universes": MIN_POS_UNIVERSES,
                 "max_targets": MAX_TARGETS},
        "targets": {c: sorted(t) for c, t in categories.items()},
    }
    (OUT_DIR / "phantomwiki_role_probe_meta.json").write_text(json.dumps(meta, indent=2))
    plot(results, OUT_DIR / "phantomwiki_role_probe.png")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
