"""Diagnostic: within-problem answer-value contrast on SVAMP variants.

The registered SVAMP answer_value probe is a cross-problem contrast, so its
pre-computation baseline (~0.62 at L40) carries topic/magnitude correlates.
Variants give the controlled instrument: operand-resampled variants of the
SAME problem have different answers, so for a target answer v, positives and
negatives share narratives — positives are correct variant rows with answer
v, negatives are correct variants of those same problems with other answers.
Prediction if the elevated floor is topic-driven: near-chance until the
computation band, late rise intact.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_svamp_variants_presence_probe as SV  # noqa: E402
import run_svamp_presence_probe as SP  # noqa: E402

LAYERS = [6, 17, 22, 30, 36, 40, 44, 52, 60, 66, 72, 76, 80]
MIN_POS = 15
MIN_GROUPS = 6


def main():
    res = pd.read_csv(SV.RESULTS_CSV)
    src = pd.read_csv(SV.SRC_CSV)
    assert len(res) == len(src)
    correct = res["original_direct_correctness"].fillna(0).astype(bool).to_numpy()
    hs = np.load(SV.HS_CACHE)["hidden_states"]
    assert hs.shape[0] == len(res)
    X = hs[correct]
    sub = src[correct].reset_index(drop=True)
    groups = sub["seed_group"].to_numpy()
    ans = sub["answer"].astype(str).to_numpy()
    prob = sub["orig_index"].to_numpy()
    n = len(sub)

    by_ans = defaultdict(list)
    for i, a in enumerate(ans):
        by_ans[a].append(i)

    tasks = []
    for v, pos_idx in sorted(by_ans.items(), key=lambda kv: -len(kv[1])):
        if len(pos_idx) < MIN_POS:
            continue
        pos_problems = set(prob[pos_idx])
        neg_idx = [i for i in range(n)
                   if prob[i] in pos_problems and ans[i] != v]
        if len(neg_idx) < MIN_POS:
            continue
        if len(np.unique(groups[pos_idx])) < MIN_GROUPS:
            continue
        y = np.zeros(n, dtype=bool); y[pos_idx] = True
        m = np.zeros(n, dtype=bool); m[pos_idx] = True; m[neg_idx] = True
        tasks.append((v, y, m))
    print(f"{len(tasks)} within-problem answer targets: "
          f"{[(v, int((y & m).sum()), int((m & ~y).sum())) for v, y, m in tasks[:10]]} ...")

    from joblib import Parallel, delayed

    def fit_masked(X_layer, y, m, g):
        return SP.fit_target_layer(X_layer[m], y[m], g[m], SP.SEED)

    results = {v: {} for v, _, _ in tasks}
    for l in LAYERS:
        X_layer = X[:, l, :].astype(np.float32)
        accs = Parallel(n_jobs=-1, verbose=0)(
            delayed(fit_masked)(X_layer, y, m, groups) for (_, y, m) in tasks)
        for (v, _, _), a in zip(tasks, accs):
            results[v][str(l)] = float(a)
        print(f"layer {l} done")

    out_path = SV.OUT_DIR / "svamp_answer_within_problem_diag.json"
    out_path.write_text(json.dumps({
        "design": "positives: correct variant rows with answer v; negatives: "
                  "correct variants of the same problems with other answers "
                  "(narrative/topic matched; GroupKFold(seed_group))",
        "layers": LAYERS,
        "per_target": results,
    }, indent=2))
    print(f"wrote {out_path}")
    arr = np.array([[results[v][str(l)] for l in LAYERS] for v in results])
    print("mean: " + " ".join(f"L{l}:{x:.2f}" for l, x in zip(LAYERS, arr.mean(axis=0))))


if __name__ == "__main__":
    main()
