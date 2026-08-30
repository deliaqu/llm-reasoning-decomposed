"""Four-stage presence probes on SVAMP (cross-benchmark generalization, R2).

Self-contained SVAMP counterpart of run_presence_probe.py. The GSM script is
hard-wired to template machinery (parse_template_entity_specs, template-
disjoint CV, pair alignment), none of which exists for SVAMP; this script
keeps the same statistical contract on SVAMP's own structure:

  - residuals at the ANSWER-POSITION (last prompt token, direct mode), all
    layers, prompts built EXACTLY as run_inference_transformers_direct.py
    builds them (chat template + TASK_CONFIG['original']['direct']);
  - per-target binary linear probes (StandardScaler + LogisticRegression),
    balanced train subsample, balanced accuracy on the test fold;
  - SEED-disjoint CV: GroupKFold on seed_group from data/svamp_probe_metadata
    .csv (the SVAMP analog of template-disjoint CV — see
    data_scripts/svamp/build_svamp_probe_metadata.py);
  - fit on the direct-correct subset only (presence-probe headline contract).

Stage -> probe categories:
  S1  entity_other   object/unit nouns present in the question (proper names
                     are structurally infeasible on SVAMP: frozen per seed)
      op_cue         arithmetic-cue words (early-decodable control)
      function_word  narrative function words (negative control)
  S2  op_presence    gold formula uses + - * / (one probe per operator)
      op_count       1-op vs 2-op formula
  S3  number         surface numeric tokens present in the question
  S4  answer_value   gold final answer equals v (one probe per frequent value)

Outputs:
  results/presence_probe/<model_short>/svamp/seed_disjoint/direct/correct/
    svamp_presence_probe.npz   per-category, per-target, per-layer accuracies
    svamp_presence_probe_meta.json
    svamp_presence_probe.png

Usage:
  python run_svamp_presence_probe.py                      # extract + fit + plot
  python run_svamp_presence_probe.py --fit_only           # reuse cached residuals
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "disentangled_evaluation"))
from instructions import TASK_CONFIG  # noqa: E402
from paths import CACHE_DIR, DATA_DIR, RESULT_DIR  # noqa: E402

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_SHORT = MODEL_NAME.split("/")[-1]

RESULTS_CSV = (
    f"{RESULT_DIR}/transformers_direct/svamp_results_{MODEL_SHORT}_transformers.csv"
)
META_CSV = f"{DATA_DIR}/svamp_probe_metadata.csv"
HS_CACHE_DIR = Path(CACHE_DIR) / "svamp_probe"
HS_CACHE = HS_CACHE_DIR / f"hidden_states_{MODEL_SHORT}_direct.npz"

OUT_DIR = (
    Path(RESULT_DIR).parent
    / "presence_probe" / MODEL_SHORT / "svamp" / "seed_disjoint" / "direct" / "correct"
)

# viability gate (mirrors the build-time audit): a target needs enough
# positives spread over enough seed clusters, and enough negatives likewise.
MIN_DOC_FREQ = 15
MIN_POS_CLUSTERS = 5
MAX_TARGETS_PER_CATEGORY = 16
# Entity categories get a relaxed cluster gate: SVAMP object nouns are frozen
# within seed families (cookies/peaches/marbles etc. live in only 2-4 clusters
# each), so pos_clusters>=5 leaves almost no decorative words. At >=3, folds
# whose test split lacks positives are skipped (fit_target_layer handles this);
# effective folds ~= the word's cluster count, and the larger K makes the
# category aggregate far more informative than the strict-gate K=2.
ENTITY_MIN_POS_CLUSTERS = 3
ENTITY_MAX_TARGETS = 24
N_FOLDS = 10
# Each (target, layer, fold) is refit with N_SEED_REPS independent balanced
# subsamples and averaged — removes single-draw subsampling jitter.
N_SEED_REPS = 5
SEED = 0

# S1 split: an object noun is OPERAND-CARRYING if >= OPERAND_ATTACH_THRESHOLD
# of its corpus occurrences directly follow one of that row's bound operand
# values ("76 dollars"); otherwise DECORATIVE. Mechanical criterion computed
# from text + symbol_binding only — never from probe results (no circularity).
# Empirical gap: books 0.29 / friends 0.22 vs people 0.50 ... packs 0.71.
OPERAND_ATTACH_THRESHOLD = 0.4

STAGE_LINES = {22: "L22", 36: "L36", 40: "L40", 80: "L80"}
CATEGORY_STYLE = {
    "entity_decorative": ("tab:blue", "S1 entity (decorative)"),
    "entity_operand": ("tab:olive", "entity (operand-carrying)"),
    "op_cue": ("tab:cyan", "S1 op-cue words"),
    "function_word": ("tab:gray", "control: function words"),
    "op_presence": ("tab:orange", "S2 formula operators"),
    "op_count": ("tab:red", "S2 op-count"),
    "number": ("tab:green", "S3 numbers (operands)"),
    "number_ansmatched": ("seagreen", "S3 numbers (answer-matched)"),
    "answer_value": ("tab:purple", "S4 answer value"),
}

# Confound control: problems containing a given small number have
# systematically smaller answers (median 5-8.5 present vs 16-19.5 absent), and
# a bare threshold on the answer value alone classifies every number target at
# 0.60-0.68 balanced acc — so once the answer becomes decodable (late layers),
# the raw `number` curve is contaminated. `number_ansmatched` refits the same
# targets with class answer-distributions equalized (quantile-bin matching in
# BOTH train and test folds), removing that axis.
N_ANSWER_MATCH_BINS = 6


# ── data assembly ─────────────────────────────────────────────────────────


def load_frames():
    res = pd.read_csv(RESULTS_CSV)
    meta = pd.read_csv(META_CSV)
    src = pd.read_csv(f"{DATA_DIR}/test_svamp.csv")
    assert len(res) == len(meta) == len(src) == 1000
    # row-index alignment was established at build time; re-assert cheaply
    assert (res["question"] == src["question"]).all(), "results/source question mismatch"
    assert (meta["answer"].astype(float) == src["answer"].astype(float)).all()
    correct = res["original_direct_correctness"].fillna(0).astype(bool).to_numpy()
    return res, meta, correct


def build_prompts(questions: list[str], tokenizer) -> list[str]:
    """EXACT mirror of run_inference_transformers_direct.generate_original."""
    instruction = TASK_CONFIG["original"]["direct"]
    prompts = []
    for question in questions:
        messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return prompts


def extract_hidden_states(questions: list[str], batch_size: int) -> np.ndarray:
    """Forward pass only (no generation); last-prompt-token residual, all layers.
    Returns (n_rows, n_layers+1, hidden) float16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, token=os.environ.get("HF_TOKEN")
    )
    tokenizer.padding_side = "left"  # left-pad => position -1 is the answer prefix
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.float16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    prompts = build_prompts(questions, tokenizer)
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        print(f"extracting {start}-{start + len(batch) - 1} / {len(prompts) - 1}")
        enc = tokenizer(
            batch, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(model.device)
        with torch.no_grad():
            # use_cache=False: no generation follows; skip the KV-cache cost.
            out = model(**enc, output_hidden_states=True, use_cache=False)
        # (n_layers+1, B, H) at the final (answer-prefix) position
        last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
        chunks.append(last.to(torch.float16).cpu().numpy())
        del out
    hs = np.concatenate(chunks, axis=0)
    HS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HS_CACHE, hidden_states=hs)
    print(f"cached hidden states {hs.shape} -> {HS_CACHE}")
    return hs


# ── target construction ───────────────────────────────────────────────────


def _select_targets(
    labels_by_target: dict,
    groups: np.ndarray,
    min_pos_clusters: int = MIN_POS_CLUSTERS,
    max_targets: int = MAX_TARGETS_PER_CATEGORY,
) -> dict:
    """Apply the viability gate and cap; labels_by_target: name -> bool array."""
    viable = {}
    for name, y in labels_by_target.items():
        y = np.asarray(y, dtype=bool)
        pos, neg = int(y.sum()), int((~y).sum())
        pos_clusters = len(np.unique(groups[y]))
        neg_clusters = len(np.unique(groups[~y]))
        if (
            min(pos, neg) >= MIN_DOC_FREQ
            and pos_clusters >= min_pos_clusters
            and neg_clusters >= min_pos_clusters
        ):
            viable[name] = (y, min(pos, neg), pos_clusters)
    ranked = sorted(viable.items(), key=lambda kv: -kv[1][1])
    return {name: y for name, (y, _, _) in ranked[:max_targets]}


def build_targets(meta: pd.DataFrame, questions: list[str], mask: np.ndarray) -> dict:
    """category -> {target_name -> bool label array over masked rows}."""
    sub = meta[mask].reset_index(drop=True)
    qs = [q for q, m in zip(questions, mask) if m]
    groups = sub["seed_group"].to_numpy()

    # degenerate preprocessed row (formula '.') -> exclude from formula-label
    # categories by treating its labels as invalid (drop from those probes)
    formula_ok = sub["op_count"].to_numpy() > 0

    def word_labels(col):
        rows = [set(json.loads(s)) for s in sub[col]]
        vocab = sorted({w for r in rows for w in r})
        return {w: np.array([w in r for r in rows]) for w in vocab}

    def operand_attachment_rate(word: str) -> float:
        """Fraction of the word's occurrences that directly follow one of the
        row's bound operand values (from symbol_binding via operand_values)."""
        occ = att = 0
        token_re = re.compile(r"\d+(?:\.\d+)?|[A-Za-z]+(?:'[a-z]+)?")
        for q, ops_json in zip(qs, sub["operand_values"]):
            ops = {str(v) for v in json.loads(ops_json)}
            ops |= {str(int(v)) for v in json.loads(ops_json)}
            toks = [t.lower() for t in token_re.findall(q)]
            for j, t in enumerate(toks):
                if t == word:
                    occ += 1
                    att += int(j > 0 and toks[j - 1] in ops)
        return att / occ if occ else 0.0

    categories: dict[str, dict] = {}
    entity_targets = _select_targets(
        word_labels("entity_other_words"), groups,
        min_pos_clusters=ENTITY_MIN_POS_CLUSTERS, max_targets=ENTITY_MAX_TARGETS,
    )
    attach = {w: operand_attachment_rate(w) for w in entity_targets}
    categories["entity_decorative"] = {
        w: y for w, y in entity_targets.items() if attach[w] < OPERAND_ATTACH_THRESHOLD
    }
    categories["entity_operand"] = {
        w: y for w, y in entity_targets.items() if attach[w] >= OPERAND_ATTACH_THRESHOLD
    }
    print("entity operand-attachment rates:",
          {w: round(r, 2) for w, r in sorted(attach.items(), key=lambda kv: kv[1])})
    categories["op_cue"] = _select_targets(word_labels("op_cue_words"), groups)
    categories["function_word"] = _select_targets(word_labels("function_words"), groups)

    ops_rows = [json.loads(s) for s in sub["operators"]]
    op_labels = {
        f"op:{op}": np.array([op in r for r in ops_rows]) & formula_ok
        for op in ["+", "-", "*", "/"]
    }
    categories["op_presence"] = _select_targets(op_labels, groups)
    categories["op_count"] = _select_targets(
        {"two_ops": (sub["op_count"].to_numpy() == 2) & formula_ok}, groups
    )

    num_rows = [set(re.findall(r"\d+", q)) for q in qs]
    num_vocab = sorted({n for r in num_rows for n in r})
    num_labels = {f"num:{n}": np.array([n in r for r in num_rows]) for n in num_vocab}
    categories["number"] = _select_targets(num_labels, groups)
    # same targets refit under answer-distribution matching (see N_ANSWER_MATCH_BINS)
    categories["number_ansmatched"] = dict(categories["number"])

    ans = sub["answer"].astype(float).to_numpy()
    ans_labels = {
        f"ans:{int(v)}": ans == v
        for v in pd.Series(ans).value_counts().head(30).index
    }
    categories["answer_value"] = _select_targets(ans_labels, groups)
    extras = {
        "entity_operand_attachment_rates": {w: round(r, 3) for w, r in attach.items()},
        "operand_attach_threshold": OPERAND_ATTACH_THRESHOLD,
    }
    return categories, groups, extras


# ── probe fitting ─────────────────────────────────────────────────────────


def _matched_subsample(idx: np.ndarray, y: np.ndarray, match_bin: np.ndarray, rng):
    """Equalize the two classes' match_bin distributions within idx: per bin,
    subsample each class to the smaller class count. Removes any axis of the
    match variable (e.g. answer magnitude) as a usable probe feature."""
    keep = []
    for b in np.unique(match_bin[idx]):
        pos = idx[(match_bin[idx] == b) & y[idx]]
        neg = idx[(match_bin[idx] == b) & ~y[idx]]
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    return np.concatenate(keep) if keep else np.array([], dtype=int)


def fit_target_layer(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int,
                     match_bin: np.ndarray | None = None):
    """Mean balanced accuracy over seed-disjoint folds, each averaged over
    N_SEED_REPS independent balanced subsample draws (removes single-draw
    jitter). With match_bin, both train AND test folds are subsampled so the
    two classes have identical match_bin distributions (confound control,
    e.g. answer value)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    accs = []
    for fold_i, (tr_full, te_full) in enumerate(folds):
        for rep in range(N_SEED_REPS):
            rng = np.random.default_rng([seed, fold_i, rep])
            tr, te = tr_full, te_full
            if match_bin is not None:
                tr = _matched_subsample(tr, y, match_bin, rng)
                te = _matched_subsample(te, y, match_bin, rng)
            y_tr = y[tr] if len(tr) else np.array([], dtype=bool)
            if y_tr.sum() < 2 or (~y_tr).sum() < 2 or len(te) < 4 or len(set(y[te])) < 2:
                break  # fold lacks both classes -> skip all reps of this fold
            pos, neg = tr[y_tr], tr[~y_tr]
            k = min(len(pos), len(neg))
            tr_bal = np.concatenate([
                rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)
            ])
            # hyperparameters mirror run_presence_probe.fit_one_layer exactly
            clf = Pipeline([
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
            ])
            clf.fit(X[tr_bal], y[tr_bal])
            accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs)) if accs else np.nan


def _fit_one_layer_all_targets(X_layer: np.ndarray, flat_targets: list, groups: np.ndarray,
                               match_bin: np.ndarray):
    """One worker = one layer: fit every target on this layer's residuals.
    Parallelism is over layers so each job ships one (n, hidden) float32 matrix
    (~28 MB) instead of duplicating it per target."""
    return [
        fit_target_layer(X_layer, y, groups, SEED,
                         match_bin=match_bin if matched else None)
        for (_, _, y, matched) in flat_targets
    ]


def fit_all(hs: np.ndarray, categories: dict, groups: np.ndarray, n_jobs: int,
            match_bin: np.ndarray, matched_categories: set):
    from joblib import Parallel, delayed

    n_layers = hs.shape[1]
    flat_targets = [
        (cat, name, y, cat in matched_categories)
        for cat, targets in categories.items() for name, y in targets.items()
    ]
    print(f"fitting {len(flat_targets) * n_layers} probes "
          f"({len(flat_targets)} targets x {n_layers} layers)")
    per_layer = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_fit_one_layer_all_targets)(
            hs[:, layer, :].astype(np.float32), flat_targets, groups, match_bin
        )
        for layer in range(n_layers)
    )
    results: dict[str, dict[str, list]] = defaultdict(dict)
    for layer, layer_accs in enumerate(per_layer):
        for (cat, name, _, _), acc in zip(flat_targets, layer_accs):
            results[cat].setdefault(name, [np.nan] * n_layers)[layer] = acc
    return {c: {n: np.array(v) for n, v in t.items()} for c, t in results.items()}


# ── plot ──────────────────────────────────────────────────────────────────


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
        ax.plot(x, mean, color=color, label=f"{label} (K={len(targets)})", lw=1.8)
        if curves.shape[0] > 1:
            # 95% bootstrap CI of the CATEGORY MEAN (resampling targets) — the
            # aggregate's uncertainty, not the per-target spread (~SD/sqrt(K)).
            K = curves.shape[0]
            boot = np.nanmean(
                curves[rng.integers(0, K, size=(2000, K))], axis=1)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    for layer, name in STAGE_LINES.items():
        ax.axvline(layer, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.text(layer, 1.02, name, ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("balanced accuracy (seed-disjoint CV)")
    ax.set_title(f"SVAMP presence probes — {MODEL_SHORT}, direct, correct-only\n"
                 "(shading: 95% bootstrap CI of category mean)", fontsize=10)
    # legend outside the axes: the decorative curve lives at ~0.5 where an
    # in-axes legend box would cover it
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"wrote {out_png}")


# ── main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--fit_only", action="store_true",
                    help="reuse cached hidden states (skip GPU extraction)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    res, meta, correct = load_frames()
    questions = res["question"].astype(str).tolist()
    print(f"direct-correct rows: {correct.sum()}/1000")

    if args.fit_only:
        hs = np.load(HS_CACHE)["hidden_states"]
        print(f"loaded cached hidden states {hs.shape}")
    else:
        hs = extract_hidden_states(questions, args.batch_size)
    assert hs.shape[0] == 1000

    categories, groups, extras = build_targets(meta, questions, correct)
    for cat, targets in categories.items():
        print(f"  {cat:18s}: {len(targets):2d} targets | {sorted(targets)[:8]}")

    ans_vals = meta[correct]["answer"].astype(float).to_numpy()
    quantiles = np.quantile(ans_vals, np.linspace(0, 1, N_ANSWER_MATCH_BINS + 1)[1:-1])
    match_bin = np.searchsorted(quantiles, ans_vals)
    results = fit_all(hs[correct], categories, groups, args.n_jobs,
                      match_bin=match_bin, matched_categories={"number_ansmatched"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "svamp_presence_probe.npz",
        **{f"{cat}|{name}": acc for cat, t in results.items() for name, acc in t.items()},
    )
    meta_out = {
        "model": MODEL_NAME,
        "results_csv": RESULTS_CSV,
        "n_correct": int(correct.sum()),
        "n_folds": N_FOLDS,
        "n_seed_reps": N_SEED_REPS,
        "ci": "95% bootstrap CI of category mean (2000 resamples over targets)",
        "cv": "GroupKFold(seed_group) — seed-disjoint",
        "gate": {"min_doc_freq": MIN_DOC_FREQ, "min_pos_clusters": MIN_POS_CLUSTERS,
                 "max_targets_per_category": MAX_TARGETS_PER_CATEGORY},
        "targets": {cat: sorted(t) for cat, t in categories.items()},
        **extras,
    }
    (OUT_DIR / "svamp_presence_probe_meta.json").write_text(json.dumps(meta_out, indent=2))
    plot(results, OUT_DIR / "svamp_presence_probe.png")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
