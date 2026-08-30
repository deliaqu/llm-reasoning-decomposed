"""Build GSM-Symbolic-style operand variants of SVAMP problems.

Each SVAMP row already carries its template form (`symbolic_question` with
w/x/y/z slots, `symbol_binding`, and the gold formula in
`symbolic_abstraction_answer`). This script re-instantiates each usable
problem with freshly sampled operand values — turning every problem into a
small template family, exactly like GSM-Symbolic's number variation.

Purpose (probe-side): the current SVAMP number probe is a cross-problem
contrast, so number presence is entangled with problem topic (12<->dozens).
Variants give the GSM-style WITHIN-template contrast: the same narrative with
and without a probed value. A fixed bank of distinctive 2-digit values is
deliberately injected (one slot per resampled variant, p=0.75) so each probed
value has balanced within-template presence.

Validity: a candidate assignment is accepted only if EVERY subexpression of
the gold formula evaluates to a non-negative integer (divisions exact, no
zero divisors) — the same constraint set the original problems satisfy.
Non-injected slots resample within the original value's digit range (2-9,
10-99, or 100-999) to stay in-distribution.

Rows per problem: variant 0 = the original values, variants 1..3 resampled.
`seed_group` is inherited from svamp_probe_metadata.csv, so GroupKFold keeps
every variant of a problem (and its whole variation cluster) in one fold.

Excluded problems: rows whose formula does not reproduce the gold answer
under the strict evaluator, or whose symbolic_question symbols do not match
the binding (≈40/1000, e.g. the known-degenerate row 554).

Output: data/test_svamp_variants.csv
Usage:  ./submit.sh data_scripts/svamp/build_svamp_variants.sh
"""

from __future__ import annotations

import ast
import json
import operator
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUT_CSV = DATA_DIR / "test_svamp_variants.csv"

N_RESAMPLED = 3          # variants per problem, in addition to the original
INJECT_PROB = 0.75       # probability a resampled variant gets one bank value
MAX_ATTEMPTS = 300
SEED = 0

# Distinctive 2-digit probe values, deliberately injected for balanced
# within-template presence contrasts.
VALUE_BANK = [17, 23, 29, 34, 38, 41, 47, 53, 59, 62, 67, 71, 76, 83, 89, 94]

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv}


def eval_checked(expr: str):
    """Evaluate an arithmetic expression; return its int value iff EVERY
    subexpression is a non-negative integer (divisions exact). Else None."""
    def rec(node):
        if isinstance(node, ast.Constant):
            v = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = rec(node.left), rec(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Div) and (right == 0 or left % right != 0):
                return None
            v = _OPS[type(node.op)](left, right)
        else:
            return None
        if isinstance(v, float):
            if v != int(v):
                return None
            v = int(v)
        if not isinstance(v, int) or v < 0:
            return None
        return v
    try:
        # .strip(): formulas carry leading/trailing spaces (" x - y "), and
        # ast.parse(mode="eval") raises IndentationError on a leading space
        return rec(ast.parse(expr.strip(), mode="eval").body)
    except (SyntaxError, ValueError, RecursionError):
        return None


def substitute(text: str, binding: dict) -> str:
    for sym, val in binding.items():
        text = re.sub(rf"\b{sym}\b", str(int(val)), text)
    return text


def parse_binding(s: str) -> dict:
    return {m.group(1): int(float(m.group(2)))
            for m in re.finditer(r"([a-z])\s*=\s*(\d+(?:\.\d+)?)", str(s))}


def digit_range(v: int):
    if v < 10:
        return 2, 9        # avoid 0/1: degenerate arithmetic (x*1, x-0)
    if v < 100:
        return 10, 99
    return 100, 999


def main():
    rng = np.random.default_rng(SEED)
    src = pd.read_csv(DATA_DIR / "test_svamp.csv")
    meta = pd.read_csv(DATA_DIR / "svamp_probe_metadata.csv")
    assert len(src) == len(meta) == 1000

    rows, excluded = [], []
    for i, r in src.iterrows():
        formula = str(r["symbolic_abstraction_answer"])
        binding = parse_binding(r["symbol_binding"])
        syms_q = set(re.findall(r"\b([a-z])\b", str(r["symbolic_question"]))) & set("wxyz")
        gold = eval_checked(substitute(formula, binding))
        if not binding or syms_q != set(binding) or gold is None \
                or gold != int(float(r["answer"])):
            excluded.append(i)
            continue

        seed_group = int(meta.iloc[i]["seed_group"])
        slots = sorted(binding)

        def emit(bind, variant_id, injected):
            rows.append({
                "question": substitute(str(r["symbolic_question"]), bind),
                "answer": eval_checked(substitute(formula, bind)),
                "orig_index": i,
                "variant_id": variant_id,
                "seed_group": seed_group,
                "injected_value": injected if injected is not None else "",
                "binding": json.dumps(bind),
                "formula": formula,
            })

        emit(binding, 0, None)  # the original values
        seen = {tuple(binding[s] for s in slots)}
        for v in range(1, N_RESAMPLED + 1):
            inject = rng.random() < INJECT_PROB
            done = False
            for _ in range(MAX_ATTEMPTS):
                cand = {}
                inj_slot = rng.choice(slots) if inject else None
                inj_val = int(rng.choice(VALUE_BANK)) if inject else None
                for s in slots:
                    if s == inj_slot:
                        cand[s] = inj_val
                    else:
                        lo, hi = digit_range(binding[s])
                        cand[s] = int(rng.integers(lo, hi + 1))
                key = tuple(cand[s] for s in slots)
                if key in seen:
                    continue
                if eval_checked(substitute(formula, cand)) is not None:
                    emit(cand, v, inj_val)
                    seen.add(key)
                    done = True
                    break
            if not done and inject:
                # constraints too tight with the forced value: retry uninjected
                for _ in range(MAX_ATTEMPTS):
                    cand = {}
                    for s in slots:
                        lo, hi = digit_range(binding[s])
                        cand[s] = int(rng.integers(lo, hi + 1))
                    key = tuple(cand[s] for s in slots)
                    if key in seen:
                        continue
                    if eval_checked(substitute(formula, cand)) is not None:
                        emit(cand, v, None)
                        seen.add(key)
                        done = True
                        break

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"excluded problems: {len(excluded)} -> usable {1000 - len(excluded)}")
    print(f"wrote {OUT_CSV}: {len(df)} rows "
          f"({(df.variant_id == 0).sum()} originals + {(df.variant_id > 0).sum()} variants)")

    # ── injection / contrast audit ────────────────────────────────────────
    print("\n=== bank-value presence audit (from question text, incl. incidental) ===")
    num_sets = [set(re.findall(r"\d+", q)) for q in df["question"]]
    groups = df["seed_group"].to_numpy()
    origs = df["orig_index"].to_numpy()
    for w in VALUE_BANK:
        y = np.array([str(w) in s for s in num_sets])
        # templates (problems) where presence VARIES across variants = the
        # within-template contrast the probe design needs
        contrast_templates = sum(
            1 for o in np.unique(origs[y])
            if 0 < y[origs == o].sum() < (origs == o).sum()
        )
        print(f"  {w}: df={y.sum():4d}  seed_groups={len(np.unique(groups[y])):2d}  "
              f"contrast_templates={contrast_templates:3d}")


if __name__ == "__main__":
    main()
