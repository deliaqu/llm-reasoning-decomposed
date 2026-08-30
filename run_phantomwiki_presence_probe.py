"""Presence probes on PhantomWiki (cross-task generalization, R2).

Counterpart of run_svamp_presence_probe.py for the PhantomWiki easy set
(24 generated universes, whole corpus in-context). Reuses that module's fit
machinery (GroupKFold + multi-seed balanced probes) with PhantomWiki-specific
prompts, targets, and CV.

CV DESIGN — the whole-corpus-in-context trap:
  Every prompt from a universe shares the SAME ~6k-token corpus, so any
  corpus-level feature (e.g. "person name X in prompt") is CONSTANT within a
  universe. Under universe-disjoint CV such targets are degenerate (they only
  decode universe identity; effective n = 24, and anchor full names span
  exactly 1 universe each — audited 0/309 cross-universe). Therefore ALL probe
  targets here are QUESTION-level features that vary within a universe and
  recur across universes:

    relation       S2 analog  relation words composing the query chain
                              (mother, friend, husband, ...); df 69-206,
                              20-24 universes each
    selector_type  S2 analog  anchor selector kind mentioned in the question
                              (occupation / hobby / date-of-birth); df 376-430
    chain_len      op-count   difficulty (1-3) as two binaries (>=2 hops,
                   analog     ==3 hops). CAVEAT: chain length correlates with
                              question length -> partially position-decodable.
    q_aggregation  qtype      "How many..." vs "Who is..." question
    answer_count   S4 analog  gold count answer value (ans:1..5), fit ONLY on
                              count-answer rows so the probe cannot reduce to
                              aggregation-vs-who classification

  Population: gen_easy_d8 rows with correct direct answers (1592 rows over 24
  universes; hard-mode dump rows are excluded — different question mix and
  guess-contamination). GroupKFold on `universe`.

  Interpretation caveat: late-layer relation decodability may partly reflect
  upcoming-answer correlates (e.g. husband-questions -> male name answers);
  early/mid-band decodability and retention are the claim-safe regions.

Outputs:
  results/presence_probe/<model>/phantomwiki/universe_disjoint/direct/correct/

Usage:
  python run_phantomwiki_presence_probe.py                 # extract + fit + plot
  python run_phantomwiki_presence_probe.py --fit_only      # reuse cached residuals
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "disentangled_evaluation"))
from instructions import TASK_CONFIG  # noqa: E402
from paths import CACHE_DIR, DATA_DIR, RESULT_DIR  # noqa: E402

# Shared fit machinery (GroupKFold, multi-seed balanced probes, bootstrap-CI
# conventions). N_FOLDS=10 / N_SEED_REPS=5 are inherited from that module.
from run_svamp_presence_probe import fit_all  # noqa: E402

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_SHORT = MODEL_NAME.split("/")[-1]

RESULTS_CSV = (
    f"{RESULT_DIR}/transformers_direct/phantomwiki_results_{MODEL_SHORT}_transformers.csv"
)
SRC_CSV = f"{DATA_DIR}/test_phantomwiki.csv"
HS_CACHE_DIR = Path(CACHE_DIR) / "phantomwiki_probe"
HS_CACHE = HS_CACHE_DIR / f"hidden_states_{MODEL_SHORT}_direct.npz"

OUT_DIR = (
    Path(RESULT_DIR).parent
    / "presence_probe" / MODEL_SHORT / "phantomwiki"
    / "universe_disjoint" / "direct" / "correct"
)

MIN_DF = 20            # min(pos, neg) document frequency
MIN_UNIVERSES = 8      # positives (and negatives) must span >= this many universes
SEED = 0

RELATIONS = [
    "mother", "father", "parent", "child", "son", "daughter", "husband",
    "wife", "brother", "sister", "sibling", "friend",
]
SELECTORS = [("occupation", r"occupation"), ("hobby", r"hobby"),
             ("dob", r"date of birth")]

STAGE_LINES = {22: "L22", 36: "L36", 40: "L40", 80: "L80"}
CATEGORY_STYLE = {
    "relation": ("tab:orange", "relation words (S2 analog)"),
    "selector_type": ("tab:cyan", "anchor selector type"),
    "chain_len": ("tab:red", "chain length (op-count analog)"),
    "q_aggregation": ("tab:gray", "aggregation vs who"),
    "answer_count": ("tab:purple", "S4 count-answer value"),
}


# ── data assembly ─────────────────────────────────────────────────────────


def load_frames():
    res = pd.read_csv(RESULTS_CSV)
    src = pd.read_csv(SRC_CSV)
    assert len(res) == len(src) == 2440
    assert (res["pw_id"] == src["pw_id"]).all(), "results/source pw_id mismatch"
    assert (res["question"] == src["question"]).all(), "results/source question mismatch"
    correct = res["original_direct_correctness"].fillna(0).astype(bool).to_numpy()
    fit_mask = (res["source"] == "gen_easy_d8").to_numpy() & correct
    return res, fit_mask


def build_prompts(questions: list[str], tokenizer) -> list[str]:
    """EXACT mirror of run_inference_transformers_direct.generate_original for
    a TEXT_ANSWER_DATASETS dataset: original_text instruction, same template."""
    instruction = TASK_CONFIG["original_text"]["direct"]
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
    """Forward pass only; last-prompt-token residual, all layers, fp16.
    batch_size default is 2: prompts are ~6-7k tokens, and out.hidden_states
    holds all positions for all 81 layers on-GPU (~9 GB per batch element)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, token=os.environ.get("HF_TOKEN")
    )
    tokenizer.padding_side = "left"
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
    print("prompt[0] tail (sanity):", repr(prompts[0][-160:]))
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        if start % (batch_size * 50) == 0:
            print(f"extracting {start}-{start + len(batch) - 1} / {len(prompts) - 1}")
        enc = tokenizer(
            batch, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(model.device)
        with torch.no_grad():
            # use_cache=False: no generation follows, and the KV cache costs
            # ~4.6 GB per batch element at these prompt lengths.
            out = model(**enc, output_hidden_states=True, use_cache=False)
        last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
        chunks.append(last.to(torch.float16).cpu().numpy())
        del out
    hs = np.concatenate(chunks, axis=0)
    HS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HS_CACHE, hidden_states=hs)
    print(f"cached hidden states {hs.shape} -> {HS_CACHE}")
    return hs


# ── target construction ───────────────────────────────────────────────────


def _viable(y: np.ndarray, universes: np.ndarray) -> bool:
    pos, neg = int(y.sum()), int((~y).sum())
    return (
        min(pos, neg) >= MIN_DF
        and len(np.unique(universes[y])) >= MIN_UNIVERSES
        and len(np.unique(universes[~y])) >= MIN_UNIVERSES
    )


def build_targets(sub: pd.DataFrame):
    """sub: the fit population (gen_easy & correct), reset_index'd.
    Returns (full_pop_categories, count_pop_categories, count_mask, universes)."""
    universes = sub["universe"].to_numpy()
    rq = sub["raw_question"].astype(str)

    full: dict[str, dict] = {}

    rel_targets = {}
    for r in RELATIONS:
        y = rq.str.contains(rf"\b{r}s?\b", case=False, regex=True).to_numpy()
        if _viable(y, universes):
            rel_targets[f"rel:{r}"] = y
    full["relation"] = rel_targets

    sel_targets = {}
    for name, pat in SELECTORS:
        y = rq.str.contains(pat, case=False, regex=True).to_numpy()
        if _viable(y, universes):
            sel_targets[f"sel:{name}"] = y
    full["selector_type"] = sel_targets

    diff = sub["difficulty"].to_numpy()
    chain = {}
    for name, y in [("hops>=2", diff >= 2), ("hops==3", diff == 3)]:
        if _viable(y, universes):
            chain[name] = y
    full["chain_len"] = chain

    agg = rq.str.startswith("How many").to_numpy()
    full["q_aggregation"] = {"how_many": agg} if _viable(agg, universes) else {}

    # S4: count-answer value, fit ONLY on count-answer rows (else the probe
    # reduces to aggregation-vs-who).
    is_count = sub["answer"].astype(str).str.fullmatch(r"\d+").to_numpy()
    ans = np.where(is_count, sub["answer"].astype(str), "")
    count_universes = universes[is_count]
    ans_targets = {}
    for v in ["1", "2", "3", "4", "5"]:
        y = (ans[is_count] == v)
        if _viable(y, count_universes):
            ans_targets[f"ans:{v}"] = y
    count_pop = {"answer_count": ans_targets}
    return full, count_pop, is_count, universes


# ── plot ──────────────────────────────────────────────────────────────────


def plot(results: dict, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
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
            ax.fill_between(x, lo, hi, color=color, alpha=0.15, lw=0)
    for layer, name in STAGE_LINES.items():
        ax.axvline(layer, color="k", ls=":", lw=0.8, alpha=0.6)
        ax.text(layer, 1.02, name, ha="center", fontsize=8,
                transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.4)
    ax.set_xlabel("layer (0 = embeddings)")
    ax.set_ylabel("balanced accuracy (universe-disjoint CV)")
    ax.set_title(f"PhantomWiki presence probes — {MODEL_SHORT}, direct, correct-only\n"
                 "(shading: 95% bootstrap CI of category mean)", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


# ── main ──────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--fit_only", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    res, fit_mask = load_frames()
    print(f"fit population: {fit_mask.sum()}/2440 (gen_easy & direct-correct), "
          f"{res[fit_mask]['universe'].nunique()} universes")

    if args.fit_only:
        hs = np.load(HS_CACHE)["hidden_states"]
        print(f"loaded cached hidden states {hs.shape}")
    else:
        hs = extract_hidden_states(res["question"].astype(str).tolist(), args.batch_size)
    assert hs.shape[0] == 2440

    sub = res[fit_mask].reset_index(drop=True)
    full, count_pop, is_count, universes = build_targets(sub)
    for cats in (full, count_pop):
        for cat, targets in cats.items():
            print(f"  {cat:14s}: {len(targets):2d} targets | {sorted(targets)}")

    hs_fit = hs[fit_mask]
    # full-population categories share the full X; answer_count fits on the
    # count-answer subpopulation (labels defined there, no agg-vs-who shortcut)
    results = fit_all(hs_fit, full, universes, args.n_jobs,
                      match_bin=np.zeros(len(sub), dtype=int), matched_categories=set())
    results_ans = fit_all(hs_fit[is_count], count_pop, universes[is_count], args.n_jobs,
                          match_bin=np.zeros(int(is_count.sum()), dtype=int),
                          matched_categories=set())
    results.update(results_ans)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / "phantomwiki_presence_probe.npz",
        **{f"{cat}|{name}": acc for cat, t in results.items() for name, acc in t.items()},
    )
    meta_out = {
        "model": MODEL_NAME,
        "results_csv": RESULTS_CSV,
        "population": "gen_easy_d8 & direct-correct",
        "n_fit": int(fit_mask.sum()),
        "cv": "GroupKFold(universe) — universe-disjoint (24 groups)",
        "gate": {"min_df": MIN_DF, "min_universes": MIN_UNIVERSES},
        "targets": {cat: sorted(t) for cat, t in results.items()},
        "caveats": [
            "corpus-level targets (person-name presence) are universe-constant and "
            "therefore infeasible: anchor full names span exactly 1 universe (0/309)",
            "chain_len correlates with question length (position-decodable component)",
            "late-layer relation decodability may reflect upcoming-answer correlates",
        ],
    }
    (OUT_DIR / "phantomwiki_presence_probe_meta.json").write_text(
        json.dumps(meta_out, indent=2))
    plot(results, OUT_DIR / "phantomwiki_presence_probe.png")
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
