"""
Operator-multiset probe — Stage-2 (Symbolic Formulation) signal.

For each template, the symbolic_abstraction_answer formula is reduced to its
operator multiset (e.g., the formula `(x*y - z*w) / r` has multiset
{'*', '*', '-', '/'}).  Templates with the same multiset form a class.
We keep only classes with at least n_folds templates, so every fold has
held-out templates from every class.

The probe is a multi-class linear classifier at each layer: predict the
multiset class from the answer-position residual.  Template-disjoint CV
splits templates within each class into n_folds chunks; train and test
templates never overlap, but classes are shared so the probe can learn
class structure from training templates and is tested on whether that
structure generalises to unseen templates.

The layer at which probe accuracy saturates indicates when the operator
structure of the formula is linearly recoverable — operationalising
"symbolic formulation done" empirically.

Usage:
    python run_op_multiset_probe.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_op_multiset_probe.py --plot_only
"""

import argparse
import ast
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "op_multiset_probe"
HS_DIR  = Path(LOGIT_LENS_RESULT_DIR).parent / "template_similarity"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULT_DIR


DATASET_NAMES = {
    "gsm_symbolic": "GSM-Symbolic",
    "gsm_p1": "GSM-P1",
    "gsm_p2": "GSM-P2",
}


# ---------------------------------------------------------------------------
# Class derivation — operator multiset from symbolic_abstraction_answer
# ---------------------------------------------------------------------------

BINOP_SYMBOLS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


def op_multiset(formula: str) -> tuple:
    """Sorted tuple of binary arithmetic operators in the formula.

    Unary signs on numeric literals, e.g. ``-120*x``, are not counted as
    subtraction operators.
    """
    if not isinstance(formula, str):
        return ()
    formula = formula.strip()
    if not formula:
        return ()

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError:
        return _op_multiset_fallback(formula)

    ops = []

    class Visitor(ast.NodeVisitor):
        def visit_BinOp(self, node):
            symbol = BINOP_SYMBOLS.get(type(node.op))
            if symbol is not None:
                ops.append(symbol)
            self.generic_visit(node)

    Visitor().visit(tree)
    return tuple(sorted(ops))


def _op_multiset_fallback(formula: str) -> tuple:
    """Best-effort operator extraction when a formula is not valid Python."""
    ops = []
    prev_nonspace = ""
    for ch in formula:
        if ch.isspace():
            continue
        if ch in "*/":
            ops.append(ch)
        elif ch in "+-":
            # Treat signs at the start of an expression, or following another
            # operator/open paren, as unary rather than binary.
            if prev_nonspace and (prev_nonspace.isalnum() or prev_nonspace in ")]}"):
                ops.append(ch)
        prev_nonspace = ch
    return tuple(sorted(ops))


def _dataset_suffix(dataset_name: str) -> str:
    return "" if dataset_name == "gsm_symbolic" else f"_{dataset_name}"


def parse_datasets(datasets: str | list[str]) -> list[str]:
    if isinstance(datasets, str):
        out = [d.strip() for d in datasets.split(",") if d.strip()]
    else:
        out = list(datasets)
    unknown = [d for d in out if d not in DATASET_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. Choices: {', '.join(DATASET_NAMES)}"
        )
    if not out:
        raise ValueError("At least one dataset is required.")
    return out


def load_result_rows(model_id: str, dataset_name: str) -> pd.DataFrame:
    """Load evaluated rows for one GSM variant."""
    file_suffix = model_id.split("/")[-1]
    files = sorted(glob.glob(
        os.path.join(RESULT_DIR, dataset_name,
                     f"{dataset_name}_set_*_results_{file_suffix}.csv")
    ))
    if not files:
        raise FileNotFoundError(
            f"No {DATASET_NAMES[dataset_name]} files for {file_suffix}"
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.copy()
    df["dataset"] = dataset_name
    return df


def template_classes(
    model_id: str,
    datasets: list[str],
    min_templates: int = 2,
    class_source: str = "all",
) -> pd.DataFrame:
    """Load GSM variants, derive op-multiset class per namespaced template.

    `template_key` is `dataset:original_id`, while `base_original_id` keeps the
    raw GSM template id for stricter cross-variant splitting.
    `class_source` chooses where the allowed class vocabulary comes from:
    "all" uses all selected datasets; otherwise use one dataset's template
    counts, then admit selected-dataset examples whose multiset is in that
    source vocabulary.
    """
    df = pd.concat(
        [load_result_rows(model_id, dataset_name) for dataset_name in datasets],
        ignore_index=True,
    )

    template_df = (
        df.drop_duplicates(subset=["dataset", "original_id"])
          [["dataset", "original_id", "symbolic_abstraction_answer"]]
          .rename(columns={"symbolic_abstraction_answer": "formula"})
          .reset_index(drop=True)
    )
    template_df["base_original_id"] = template_df["original_id"].astype(int)
    template_df["template_key"] = (
        template_df["dataset"].astype(str) + ":" +
        template_df["base_original_id"].astype(str)
    )
    template_df["multiset"] = template_df["formula"].apply(op_multiset)
    multiset_counts = Counter(template_df["multiset"].tolist())
    template_df["multiset_size"] = template_df["multiset"].apply(
        lambda m: multiset_counts[m]
    )

    if class_source == "all":
        source_df = template_df
    else:
        if class_source not in DATASET_NAMES:
            raise ValueError(
                f"Unknown class_source '{class_source}'. Use 'all' or one of: "
                f"{', '.join(DATASET_NAMES)}"
            )
        if class_source not in datasets:
            raise ValueError(
                f"class_source='{class_source}' must also be present in --datasets "
                f"({datasets})."
            )
        source_df = template_df[template_df["dataset"] == class_source]

    source_counts = Counter(source_df["multiset"].tolist())
    # Assign integer class_id only to multisets with enough templates for CV;
    # singletons get class_id = -1 (excluded from probing).
    shareable = sorted(
        [m for m, c in source_counts.items() if c >= min_templates],
        key=lambda m: (-source_counts[m], m),
    )
    multiset_to_class = {m: i for i, m in enumerate(shareable)}
    template_df["class_id"] = template_df["multiset"].apply(
        lambda m: multiset_to_class.get(m, -1)
    )
    return template_df, shareable


def _hidden_cache_paths(model_short: str, dataset_name: str):
    # Delegate to run_template_similarity so any per-dataset CACHE_ROOT_OVERRIDES
    # for large paired datasets are honored.
    from run_template_similarity import cache_path, meta_path
    return (
        cache_path(model_short, correct_only=False,
                   dataset_name=dataset_name, mode="direct"),
        meta_path(model_short, correct_only=False,
                  dataset_name=dataset_name, mode="direct"),
    )


def load_aligned_dataset(model_id: str, model_short: str, dataset_name: str):
    """Load one dataset's cached hidden states and align metadata to result rows."""
    hs_path, meta_path = _hidden_cache_paths(model_short, dataset_name)
    if not hs_path.exists():
        raise FileNotFoundError(
            f"Cached hidden states not found at {hs_path}. "
            f"Run run_template_similarity.py --dataset {dataset_name} first."
        )
    hidden = np.load(hs_path)
    meta = pd.read_csv(meta_path)
    if hidden.shape[0] != len(meta):
        raise ValueError(
            f"Hidden-state row count mismatch for {dataset_name}: "
            f"{hs_path} has {hidden.shape[0]} rows, metadata has {len(meta)}."
        )
    rows = load_result_rows(model_id, dataset_name)
    df_aligned = (
        meta[["original_id", "instance"]]
        .merge(rows, on=["original_id", "instance"], how="left")
        .reset_index(drop=True)
    )
    if len(df_aligned) != len(meta) or df_aligned["question"].isna().any():
        raise ValueError(f"Could not align {dataset_name} results to {meta_path}.")
    df_aligned["dataset"] = dataset_name
    df_aligned["base_original_id"] = df_aligned["original_id"].astype(int)
    df_aligned["template_key"] = (
        df_aligned["dataset"].astype(str) + ":" +
        df_aligned["base_original_id"].astype(str)
    )
    return hidden, df_aligned


# ---------------------------------------------------------------------------
# Template-disjoint multi-class CV
# ---------------------------------------------------------------------------

def template_disjoint_multiclass_splits(
    template_ids: np.ndarray,
    template_class_id: dict,
    template_split_group: dict,
    n_folds: int,
    seed: int,
):
    """Within each class, partition split groups round-robin into folds.
    Fold i's test templates are chunk i across all classes; train is the rest.

    `template_split_group` controls the leakage policy:
      - variant-disjoint: group is dataset:original_id
      - base-disjoint: group is original_id shared across GSM variants
    """
    rng = np.random.RandomState(seed)

    class_to_groups: dict = defaultdict(set)
    group_to_templates: dict = defaultdict(set)
    for t in template_ids:
        t = str(t)
        cid = template_class_id.get(t)
        if cid is None or cid < 0:
            continue  # singleton class → not used
        group = template_split_group[t]
        class_to_groups[cid].add(group)
        group_to_templates[group].add(t)

    # Round-robin split per class
    class_chunks: dict = {}
    group_to_fold = {}
    for cid, groups in class_to_groups.items():
        groups = sorted(groups)
        rng.shuffle(groups)
        chunks = [[] for _ in range(n_folds)]
        for i, group in enumerate(groups):
            if group not in group_to_fold:
                group_to_fold[group] = i % n_folds
            chunks[group_to_fold[group]].extend(group_to_templates[group])
        class_chunks[cid] = chunks

    splits = []
    for fold in range(n_folds):
        test_t  = set()
        train_t = set()
        for cid, chunks in class_chunks.items():
            test_t.update(chunks[fold])
            for j, ch in enumerate(chunks):
                if j != fold:
                    train_t.update(ch)
        splits.append((train_t, test_t))
    return splits


def fit_multiclass_layer(X_tr, y_tr, X_te, y_te, C=1.0, max_iter=2000,
                         class_weight="balanced"):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(
            solver="lbfgs", max_iter=max_iter, C=C,
            class_weight=class_weight,
        )),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.score(X_te, y_te)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    accs: np.ndarray,           # (n_layers, n_folds)
    chance_majority: float,
    chance_uniform: float,
    n_classes: int,
    n_templates: int,
    model_short: str,
    out_dir: Path,
):
    n_layers = accs.shape[0]
    layers = np.arange(n_layers)
    mean = np.nanmean(accs, axis=1)
    se   = np.nanstd(accs, axis=1) / np.sqrt(np.sum(~np.isnan(accs), axis=1).clip(min=1))

    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    fig.suptitle(
        f"Operator-multiset probe (Stage-2 Formulation signal)\n"
        f"{model_short}  —  {n_classes} classes, {n_templates} templates",
        fontsize=12,
    )
    ax.fill_between(layers, mean - 2*se, mean + 2*se, alpha=0.20, color="#1f77b4")
    ax.plot(
        layers, mean, color="#1f77b4", lw=2.0,
        label=f"Probe accuracy ({accs.shape[1]}-fold CV mean)",
    )
    ax.axhline(chance_majority, color="grey", lw=0.8, ls="--",
               label=f"Majority-class baseline ({chance_majority:.3f})")
    ax.axhline(chance_uniform, color="grey", lw=0.8, ls=":",
               label=f"Uniform-class baseline ({chance_uniform:.3f})")
    ax.axvspan(21, 24, alpha=0.10, color="#fdae61", label="Abstraction (L21–24)")
    ax.axvspan(25, 38, alpha=0.10, color="#abd9e9", label="Formulation (L25–38)")
    ax.axvspan(39, 41, alpha=0.18, color="#d7191c", label="Binding cliff (L39–40)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plot_path = out_dir / f"op_multiset_probe_{model_short}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model_id: str, n_folds: int, n_jobs: int, seed: int,
        correct_only: bool = True,
        correctness_field: str = "symbolic_abstraction_cot_correctness",
        min_templates: int = 2,
        C: float = 0.0001,
        datasets: str | list[str] = "gsm_symbolic",
        split_mode: str = "variant-disjoint",
        class_source: str = "all"):
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    datasets = parse_datasets(datasets)
    if split_mode not in {"variant-disjoint", "base-disjoint"}:
        raise ValueError("split_mode must be 'variant-disjoint' or 'base-disjoint'")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    dataset_tag = "+".join(datasets)
    result_tag = model_short
    if (
        datasets != ["gsm_symbolic"]
        or split_mode != "variant-disjoint"
        or class_source != "all"
    ):
        result_tag = f"{model_short}_{dataset_tag}_{split_mode}_classes-{class_source}"

    # ── Load class structure ──────────────────────────────────────────────────
    template_df, shareable_multisets = template_classes(
        model_id,
        datasets=datasets,
        min_templates=min_templates,
        class_source=class_source,
    )
    template_class_id = dict(zip(template_df["template_key"], template_df["class_id"]))
    template_base_id = dict(zip(template_df["template_key"], template_df["base_original_id"]))
    if split_mode == "variant-disjoint":
        template_split_group = {t: t for t in template_class_id}
    else:
        template_split_group = {
            t: f"base:{template_base_id[t]}" for t in template_class_id
        }

    n_templates_total = len(template_df)
    template_df_in_class = template_df[template_df["class_id"] >= 0]
    n_classes = template_df_in_class["class_id"].nunique()
    n_templates_in_class = len(template_df_in_class)
    print(f"Templates total: {n_templates_total}")
    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 operator-multiset classes with >= {min_templates} "
            f"templates each; found {n_classes}."
        )
    print(f"Probe config: C={C}, min_templates={min_templates}, "
          f"correctness_field='{correctness_field}', "
          f"datasets={datasets}, split_mode={split_mode}, "
          f"class_source={class_source}")
    print(f"Shareable classes (≥ {min_templates} templates each): {n_classes}")
    print(f"Templates in shareable classes: {n_templates_in_class}")
    print()
    print("Class population (top 15):")
    for cid in sorted(template_df_in_class["class_id"].unique()):
        rows = template_df_in_class[template_df_in_class["class_id"] == cid]
        ms = shareable_multisets[cid]
        print(f"  class {cid:>3d}  ({len(rows)} templates)  multiset={''.join(ms)}")

    # ── Load cached residuals ────────────────────────────────────────────────
    hidden_parts = []
    df_parts = []
    for dataset_name in datasets:
        hidden_i, df_i = load_aligned_dataset(model_id, model_short, dataset_name)
        hidden_parts.append(hidden_i)
        df_parts.append(df_i)
        print(f"\n{DATASET_NAMES[dataset_name]} hidden states: {hidden_i.shape}")
    hidden = np.concatenate(hidden_parts, axis=0)
    df = pd.concat(df_parts, ignore_index=True)
    print(f"Combined hidden states: {hidden.shape}")
    template_ids = df["template_key"].astype(str).values

    # Filter 1: instance is in a shareable class
    in_shareable = np.array(
        [template_class_id.get(str(t), -1) >= 0 for t in template_ids],
        dtype=bool,
    )
    print(f"Instances in shareable classes: {in_shareable.sum()}/{len(template_ids)}")

    # Filter 2 (optional): the model produced the correct symbolic abstraction
    # for that instance.  Default uses COT correctness — the broader filter
    # that captures "the model's representation of the question is correct"
    # even when the direct format mangles output.  Direct correctness is
    # narrower (requires the residual at answer-position-end to directly
    # carry enough to output correctly).
    if correct_only:
        if correctness_field not in df.columns:
            raise ValueError(
                f"Column '{correctness_field}' not in aligned dataframe; "
                "available: " + ", ".join(c for c in df.columns
                                          if "correctness" in c)
            )
        sym_correct = df[correctness_field].map(
            lambda x: str(x).strip().lower() in {"true", "1"}
            if not isinstance(x, bool) else x
        ).values
        keep_mask = in_shareable & sym_correct
        n_dropped = in_shareable.sum() - keep_mask.sum()
        print(f"Filter '{correctness_field}': dropped {n_dropped} wrong-formulation "
              f"instances; {keep_mask.sum()} remain.")
    else:
        keep_mask = in_shareable

    keep_idx = np.where(keep_mask)[0]
    keep_template_ids = template_ids[keep_mask]
    keep_class_ids = np.array(
        [template_class_id[str(t)] for t in keep_template_ids],
        dtype=np.int64,
    )

    # After correctness filtering, some templates may have lost all instances
    # or fallen below n_folds for their class. Re-check class viability.
    if correct_only:
        templates_remaining = {str(t) for t in keep_template_ids.tolist()}
        class_to_groups: dict = defaultdict(set)
        for t in templates_remaining:
            cid = template_class_id.get(t, -1)
            if cid >= 0:
                class_to_groups[cid].add(template_split_group[t])
        viable_classes = {
            cid for cid, groups in class_to_groups.items() if len(groups) >= min_templates
        }
        if len(viable_classes) < 2:
            raise ValueError(
                f"After correctness filter, only {len(viable_classes)} classes "
                f"have >= {min_templates} split groups with any correct instances. "
                "Consider lowering --min_templates or running without --correct_only."
            )
        # Re-mask to keep only instances whose class is still viable
        viable_mask = np.array(
            [template_class_id.get(str(t), -1) in viable_classes for t in keep_template_ids],
            dtype=bool,
        )
        keep_idx = keep_idx[viable_mask]
        keep_template_ids = keep_template_ids[viable_mask]
        keep_class_ids = keep_class_ids[viable_mask]
        print(f"Viable classes after re-check: {len(viable_classes)}; "
              f"final kept instances: {len(keep_idx)}")
        # Reduce shareable_multisets / n_classes to match viable_classes for plotting/baselines
        n_classes = len(viable_classes)
    print(f"\nFinal kept: {len(keep_idx)} instances "
          f"across {len(set(keep_template_ids.tolist()))} templates "
          f"and {len({template_split_group[str(t)] for t in keep_template_ids.tolist()})} "
          f"split groups")

    # ── Compute chance baselines ─────────────────────────────────────────────
    label_counts = Counter(keep_class_ids.tolist())
    n_total = len(keep_class_ids)
    majority_baseline = max(label_counts.values()) / n_total
    uniform_baseline = 1.0 / n_classes
    print(f"Majority-class baseline: {majority_baseline:.4f}")
    print(f"Uniform-class baseline:  {uniform_baseline:.4f}")

    # ── Build template-disjoint folds (templates only) ──────────────────────
    splits_t = template_disjoint_multiclass_splits(
        keep_template_ids, template_class_id, template_split_group, n_folds, seed,
    )
    fold_idx_pairs = []
    for fold, (train_t, test_t) in enumerate(splits_t):
        in_train = np.isin(keep_template_ids, list(train_t))
        in_test  = np.isin(keep_template_ids, list(test_t))
        train_idx = keep_idx[in_train]
        test_idx  = keep_idx[in_test]
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(
                f"Fold {fold} is empty: {len(train_idx)} train instances, "
                f"{len(test_idx)} test instances. Try lowering --n_folds."
            )
        # Sanity: train and test split groups are disjoint.
        train_groups = {template_split_group[str(t)] for t in template_ids[train_idx]}
        test_groups = {template_split_group[str(t)] for t in template_ids[test_idx]}
        assert not (train_groups & test_groups), f"Fold {fold}: train/test group overlap!"
        fold_idx_pairs.append((train_idx, test_idx))
        print(f"  Fold {fold}: train_templates={len(train_t)} "
              f"({len(train_idx)} instances)  "
              f"test_templates={len(test_t)} ({len(test_idx)} instances)  "
              f"train_groups={len(train_groups)} test_groups={len(test_groups)}")

    # ── Train multi-class probes per layer ───────────────────────────────────
    n_layers = hidden.shape[1]

    def _layer_acc(L):
        accs = []
        for train_idx, test_idx in fold_idx_pairs:
            X_tr = hidden[train_idx, L, :].astype(np.float32)
            y_tr = np.array(
                [template_class_id[str(t)] for t in template_ids[train_idx]],
                dtype=np.int64,
            )
            X_te = hidden[test_idx, L, :].astype(np.float32)
            y_te = np.array(
                [template_class_id[str(t)] for t in template_ids[test_idx]],
                dtype=np.int64,
            )
            if len(np.unique(y_tr)) < 2:
                raise ValueError(
                    f"Layer {L}: training fold has fewer than 2 classes."
                )
            accs.append(fit_multiclass_layer(X_tr, y_tr, X_te, y_te, C=C))
        return accs

    # Parallelise over layers
    print(f"\nFitting {n_layers} layers × {n_folds} folds…")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_layer_acc)(L) for L in tqdm(range(n_layers), desc="Layers")
    )
    accs = np.array(results, dtype=np.float32)  # (n_layers, n_folds)

    # ── Save and summarise ──────────────────────────────────────────────────
    np.savez(
        OUT_DIR / f"op_multiset_probe_{result_tag}.npz",
        accs=accs,
        majority_baseline=majority_baseline,
        uniform_baseline=uniform_baseline,
    )
    with open(OUT_DIR / f"op_multiset_probe_{result_tag}_meta.json", "w") as f:
        json.dump({
            "datasets": datasets,
            "split_mode": split_mode,
            "class_source": class_source,
            "n_classes": n_classes,
            "n_templates": n_templates_in_class,
            "n_templates_total": n_templates_total,
            "n_folds": n_folds,
            "majority_baseline": float(majority_baseline),
            "uniform_baseline": float(uniform_baseline),
            "shareable_multisets": [list(m) for m in shareable_multisets],
        }, f, indent=2)

    mean = accs.mean(axis=1)
    print("\nProbe accuracy by layer:")
    print(f"{'L':>4}  {'mean':>7}" + "".join(
        f"  {f'fold{f}':>7}" for f in range(n_folds)
    ))
    for L in [0, 5, 10, 15, 20, 21, 24, 25, 30, 35, 38, 39, 40, 50, 70, 79, 80]:
        if L >= n_layers:
            continue
        line = f"{L:>4}  {mean[L]:>7.4f}"
        for f in range(n_folds):
            line += f"  {accs[L, f]:>7.4f}"
        print(line)

    plot_results(
        accs, majority_baseline, uniform_baseline,
        n_classes, n_templates_in_class, result_tag, OUT_DIR,
    )


def _result_tag(
    model_id: str,
    datasets: list[str],
    split_mode: str,
    class_source: str = "all",
) -> str:
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    if (
        datasets == ["gsm_symbolic"]
        and split_mode == "variant-disjoint"
        and class_source == "all"
    ):
        return model_short
    return f"{model_short}_{'+'.join(datasets)}_{split_mode}_classes-{class_source}"


def plot_only(
    model_id: str,
    datasets: str | list[str] = "gsm_symbolic",
    split_mode: str = "variant-disjoint",
    class_source: str = "all",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = parse_datasets(datasets)
    result_tag = _result_tag(model_id, datasets, split_mode, class_source)
    npz_path = OUT_DIR / f"op_multiset_probe_{result_tag}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No results at {npz_path}")
    data = np.load(npz_path)
    with open(OUT_DIR / f"op_multiset_probe_{result_tag}_meta.json") as f:
        meta = json.load(f)
    plot_results(
        data["accs"],
        float(data["majority_baseline"]), float(data["uniform_baseline"]),
        meta["n_classes"], meta["n_templates"], result_tag, OUT_DIR,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--datasets",
        default="gsm_symbolic",
        help="Comma-separated GSM variants: gsm_symbolic,gsm_p1,gsm_p2",
    )
    parser.add_argument(
        "--split_mode",
        default="variant-disjoint",
        choices=["variant-disjoint", "base-disjoint"],
        help="Hold out dataset:original_id templates, or raw original_id families.",
    )
    parser.add_argument(
        "--class_source",
        default="all",
        help="Class vocabulary source: 'all' or one selected dataset such as gsm_symbolic.",
    )
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=0.0001,
                        help="L2 regularization strength (smaller = stronger).")
    parser.add_argument("--min_templates", type=int, default=2,
                        help="Minimum templates per class to be probed.")
    parser.add_argument(
        "--correctness_field",
        default="symbolic_abstraction_cot_correctness",
        choices=[
            "symbolic_abstraction_direct_correctness",
            "symbolic_abstraction_cot_correctness",
        ],
        help="Which correctness column to filter by when correct_only is on. "
             "COT (default) is the broader filter; direct is narrower.",
    )
    parser.add_argument(
        "--all_instances", action="store_true",
        help="Use all instances regardless of symbolic_abstraction correctness "
             "(default: restrict to correct-formulation instances).",
    )
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_only(
            args.model_id,
            datasets=args.datasets,
            split_mode=args.split_mode,
            class_source=args.class_source,
        )
    else:
        run(
            args.model_id, args.n_folds, args.n_jobs, args.seed,
            correct_only=not args.all_instances,
            correctness_field=args.correctness_field,
            min_templates=args.min_templates,
            C=args.C,
            datasets=args.datasets,
            split_mode=args.split_mode,
            class_source=args.class_source,
        )


if __name__ == "__main__":
    main()
