"""Diagnostic: adjacency-matched role contrast for chain intermediates.

The registered intermediate targets contrast on-chain names against
name-absent rows, so their pre-binding baseline carries anchor-neighborhood
aggregation (intermediates live in the anchor's article) on top of presence.
This refit matches that out: positives = rows where the name is a chain
intermediate AND occurs in the anchor's article; negatives = rows where the
same name is an uninvolved (bystander, non-inserted) corpus name AND occurs
in the anchor's article. Presence, adjacency, and copy identity are equal
across classes; only chain membership differs. Prediction under the
adjacency account: baseline near chance before the binding step at L39.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_phantomwiki_role_probe as R  # noqa: E402
import run_svamp_presence_probe as SP  # noqa: E402

LAYERS = [6, 17, 22, 30, 36, 38, 39, 40, 44, 60, 70, 80]
MIN_SIDE = 15
MIN_UNIVERSES = 6


def main():
    res, correct = R.load_frames()
    hs = np.load(R.HS_CACHE)["hidden_states"]
    X = hs[correct]
    sub = res[correct].reset_index(drop=True)
    groups = sub["universe"].to_numpy()
    n = len(sub)

    inter_adj = defaultdict(list)  # name -> row idx where intermediate & in anchor article
    byst_adj = defaultdict(list)
    for i, row in enumerate(sub.itertuples()):
        ctx = row.question.split("\n\nQuestion:")[0]
        parts = re.split(r"\n(?=# )", ctx)
        arts = {p.splitlines()[0].lstrip("# ").strip(): p
                for p in parts if p.startswith("# ")}
        anchors = json.loads(row.role_anchor)
        anchor_art = None
        for title, text in arts.items():
            if any(title.split()[0] == a for a in anchors):
                anchor_art = text
                break
        if anchor_art is None:
            continue
        inters = set(json.loads(row.role_intermediate))
        bysts = (set(json.loads(row.role_bystander))
                 - set(json.loads(row.inserted_names)) - inters)
        for w in inters:
            if re.search(rf"\b{re.escape(w)}\b", anchor_art):
                inter_adj[w].append(i)
        for w in bysts:
            if re.search(rf"\b{re.escape(w)}\b", anchor_art):
                byst_adj[w].append(i)

    tasks = []
    for w in sorted(inter_adj, key=lambda k: -len(inter_adj[k])):
        pos, neg = inter_adj[w], byst_adj.get(w, [])
        if len(pos) < MIN_SIDE or len(neg) < MIN_SIDE:
            continue
        if len(np.unique(groups[pos])) < MIN_UNIVERSES:
            continue
        y = np.zeros(n, dtype=bool)
        y[pos] = True
        m = np.zeros(n, dtype=bool)
        m[pos] = True
        m[neg] = True
        tasks.append((w, y, m))
    print(f"{len(tasks)} adjacency-matched targets: "
          f"{[(w, int((y & m).sum()), int((m & ~y).sum())) for w, y, m in tasks]}")

    from joblib import Parallel, delayed

    def fit_masked(X_layer, y, m, g):
        return SP.fit_target_layer(X_layer[m], y[m], g[m], SP.SEED)

    results = {w: {} for w, _, _ in tasks}
    for l in LAYERS:
        X_layer = X[:, l, :].astype(np.float32)
        accs = Parallel(n_jobs=-1, verbose=0)(
            delayed(fit_masked)(X_layer, y, m, groups) for (_, y, m) in tasks)
        for (w, _, _), a in zip(tasks, accs):
            results[w][str(l)] = float(a)
        print(f"layer {l} done")

    out = {
        "design": "positives: name is chain intermediate AND in anchor article; "
                  "negatives: same name uninvolved AND in anchor article "
                  "(presence/adjacency/copy matched; only chain membership differs)",
        "layers": LAYERS,
        "per_target": results,
    }
    out_path = R.OUT_DIR / "phantomwiki_role_probe_adjmatched_diag.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    arr = np.array([[results[w][str(l)] for l in LAYERS] for w in results])
    print("mean: " + " ".join(f"L{l}:{v:.2f}" for l, v in zip(LAYERS, arr.mean(axis=0))))


if __name__ == "__main__":
    main()
