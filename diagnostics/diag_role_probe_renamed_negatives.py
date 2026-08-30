"""Diagnostic: quantify the copy-identity shortcut in the role-probe floor.

All 90 name targets are bank names, so their positives exist only in renamed
copies while ~56% of masked negatives are copy-0 rows — a pure renamed-copy
detector scores 0.78 balanced accuracy without any name information. This
refit restricts every name-category mask to renamed rows (copy_id > 0), which
makes copy identity useless to the probe. Prediction under the shortcut
account: the bystander floor collapses toward the genuine presence signal;
role-vs-floor differences and trajectory shapes are unchanged.

Reads the merged hidden-state cache; fits only the pre-registered readout
layers. Writes a JSON next to the registered outputs (separate file; the
registered npz is untouched).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
import run_phantomwiki_role_probe as R  # noqa: E402
import run_svamp_presence_probe as SP  # noqa: E402

LAYERS = [6, 17, 22, 36, 40, 70, 80]
NAME_CATS = ["name_anchor", "name_intermediate", "name_answer",
             "name_bystander", "name_inserted"]
MIN_SIDE = 15


def main():
    res, correct = R.load_frames()
    hs = np.load(R.HS_CACHE)["hidden_states"]
    assert hs.shape[0] == len(res)
    X = hs[correct]
    sub = res[correct].reset_index(drop=True)
    cats, groups = R.build_targets(sub)
    renamed = (sub["copy_id"] > 0).to_numpy()

    tasks, skipped = [], []
    for cat in NAME_CATS:
        for tname, (y, mask) in cats[cat].items():
            m2 = mask & renamed
            pos, neg = int((y & m2).sum()), int((m2 & ~y).sum())
            if min(pos, neg) < MIN_SIDE:
                skipped.append((tname, pos, neg))
                continue
            tasks.append((cat, tname, y, m2))
    print(f"{len(tasks)} targets viable under renamed-only negatives; "
          f"skipped {len(skipped)}: {skipped}")

    from joblib import Parallel, delayed

    def fit_masked(X_layer, y, m, g):
        return SP.fit_target_layer(X_layer[m], y[m], g[m], SP.SEED)

    results = {f"{c}|{t}": {} for c, t, _, _ in tasks}
    for l in LAYERS:
        X_layer = X[:, l, :].astype(np.float32)
        accs = Parallel(n_jobs=-1, verbose=1)(
            delayed(fit_masked)(X_layer, y, m, groups) for (_, _, y, m) in tasks
        )
        for (c, t, _, _), a in zip(tasks, accs):
            results[f"{c}|{t}"][str(l)] = float(a)
        print(f"layer {l} done")

    out = {
        "design": "name-category masks restricted to copy_id>0 (negatives lose "
                  "all copy-0 rows; positives unchanged — all were renamed)",
        "layers": LAYERS,
        "skipped_targets": skipped,
        "per_target": results,
    }
    out_path = R.OUT_DIR / "phantomwiki_role_probe_renamedneg_diag.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    for cat in NAME_CATS:
        arr = np.array([[results[k][str(l)] for l in LAYERS]
                        for k in results if k.startswith(cat + "|")])
        print(f"{cat:18s}: " + " ".join(
            f"L{l}:{m:.2f}" for l, m in zip(LAYERS, arr.mean(axis=0))))


if __name__ == "__main__":
    main()
