"""Build data/test_phantomwiki.csv from the pre-generated PhantomWiki v1 dump.

Cross-task generalization candidate (R2): kinship/attribute multi-hop reasoning
over small fictional universes — the modern, contamination-free CLUTRR
descendant (kilian-group/phantom-wiki-v1, ICML 2025).

Design:
  - Whole corpus in-context: each row's `question` column = all articles of
    that universe + the raw question, so the pipeline's single-prompt format
    applies unchanged (~6k tokens).
  - Two universe sources:
      seeds 1-3   pre-generated v1 dump parquets (size-50, depth-20 HARD mode;
                  difficulty ladder 1-23) — kept for the accuracy-x-difficulty
                  analysis.
      seeds 4-27  locally generated EASY universes (question-depth 8,
                  --easy-mode; difficulty 1-3 only) — the probe population;
                  see generate_universes.sh. Hard-mode "correct" rows at d>=7
                  are guess-contaminated (count answers: mode-guessing scores
                  0.37), hence the easy regeneration.
  - SINGLE-answer questions only (multi-answer rows can't be graded by ####
    exact-match and are unusable as probe targets).
  - `difficulty` (reasoning hops) and `qtype` are carried through so slices
    cut at analysis time without re-running inference.
  - `universe` (seed) is the CV grouping unit (universe-disjoint folds).

Source parquets are downloaded beforehand to data/phantomwiki/ (see repo:
huggingface.co/datasets/kilian-group/phantom-wiki-v1).

Usage:
  python data_scripts/phantomwiki/build_phantomwiki_dataset.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PW_DIR = PROJECT_ROOT / "data" / "phantomwiki"
GEN_DIR = PW_DIR / "generated"
OUT_CSV = PROJECT_ROOT / "data" / "test_phantomwiki.csv"

SEEDS = (1, 2, 3)
GEN_SEEDS = tuple(range(4, 28))

PROMPT_HEADER = (
    "Read the following articles about a fictional universe, then answer the "
    "question using only the information in the articles.\n\n"
)


def build_universe_context(seed: int) -> str:
    tc = pd.read_parquet(PW_DIR / f"text-corpus_size50_seed{seed}.parquet")
    # fixed corpus order as shipped; articles already carry their own titles
    return "\n\n".join(a.strip() for a in tc["article"].tolist())


def rows_from_qa(qa_records, context: str, seed: int, source: str):
    rows = []
    for r in qa_records:
        if len(r["answer"]) != 1:
            continue
        rows.append({
            "question": f"{PROMPT_HEADER}{context}\n\nQuestion: {r['question']}",
            "answer": str(r["answer"][0]),
            "pw_id": r["id"],
            "universe": seed,
            "raw_question": r["question"],
            "difficulty": int(r["difficulty"]),
            "qtype": int(r["type"]),
            "source": source,
        })
    return rows


def main():
    rows = []
    for seed in SEEDS:
        context = build_universe_context(seed)
        qa = pd.read_parquet(PW_DIR / f"question-answer_size50_seed{seed}.parquet")
        new = rows_from_qa(qa.to_dict("records"), context, seed, "dump_hard_d20")
        print(f"seed {seed:2d} (dump): {len(new)}/{len(qa)} single-answer | "
              f"context ~{len(context.split())} words")
        rows.extend(new)
    for seed in GEN_SEEDS:
        gen = GEN_DIR / f"seed_{seed}"
        arts = json.load(open(gen / "articles.json"))
        qa = json.load(open(gen / "questions.json"))
        # config signature guard: easy-mode depth-8 universes only
        diffs = {q["difficulty"] for q in qa}
        assert len(qa) == 140 and max(diffs) <= 3 and len(arts) >= 40, \
            f"seed {seed}: unexpected generation config (n={len(qa)}, diffs={sorted(diffs)})"
        context = "\n\n".join(a["article"].strip() for a in arts)
        new = rows_from_qa(qa, context, seed, "gen_easy_d8")
        print(f"seed {seed:2d} (gen) : {len(new)}/{len(qa)} single-answer | "
              f"context ~{len(context.split())} words")
        rows.extend(new)
    df = pd.DataFrame(rows)
    assert df["pw_id"].is_unique
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}: {len(df)} rows")
    print("difficulty histogram:", df["difficulty"].value_counts().sort_index().to_dict())
    print("answers look like:", df["answer"].sample(8, random_state=0).tolist())


if __name__ == "__main__":
    main()
