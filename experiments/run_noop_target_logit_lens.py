"""
Logit lens focused on noop distractors and relevant values.

Tracks, at each layer, the logits for:
- the gold answer token
- the model's corrupted/wrong answer token
- the primary distractor value from the reviewed noop clause
- the computation-relevant values used by the underlying symbolic problem

This is designed to answer questions like:
- when does the model start preferring the gold answer over the distractor?
- when does the model favor relevant values over the noop distractor?

Usage:
    python run_noop_target_logit_lens.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_noop_target_logit_lens.py --model_id meta-llama/Llama-3.3-70B-Instruct --plot_only
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    ANSWER_PREFIX,
    COT_INSTRUCTION,
    DATA_DIR,
    HOME_DIR,
    DIRECT_INSTRUCTION,
    LOGIT_LENS_RESULT_DIR,
    MODEL_NAME_MAP,
)
from utils.activation_patching_utils import tokenize
from utils.logit_lens_utils import forward_with_hooks, get_logit_lens, load_translators
from utils.noop_utils import (
    DATASET_CHOICES,
    annotate_noop_relevant_values,
    annotate_primary_noop_distractor,
    load_model,
    normalize_numeric_value,
    to_serializable,
)

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR)
NOOP_SOURCE_DIR = Path(HOME_DIR) / "data" / "gsm_symbolic" / "split_datasets_noop"
GROUP_DATASETS = {
    "noop_correct": (
        "noop_correct_sym_abs_correct",
        "noop_correct_sym_abs_wrong",
    ),
    "noop_wrong": (
        "noop_wrong_sym_abs_correct",
        "noop_wrong_sym_abs_wrong",
    ),
    "cot_noop_correct": (
        "cot_noop_correct_sym_abs_correct",
        "cot_noop_correct_sym_abs_wrong",
    ),
    "cot_noop_wrong": (
        "cot_noop_wrong_sym_abs_correct",
        "cot_noop_wrong_sym_abs_wrong",
    ),
}


def _dataset_prompt_mode(dataset):
    if dataset in GROUP_DATASETS:
        dataset = GROUP_DATASETS[dataset][0]
    return "direct" if dataset.startswith("direct") or dataset.startswith("noop_") else "cot"


def _parse_maybe_literal(value):
    if isinstance(value, (dict, list)):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _load_noop_source_annotations():
    frames = []
    for path in sorted(NOOP_SOURCE_DIR.glob("test_gsm_noop_set_*.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = annotate_noop_relevant_values(df)
    df = annotate_primary_noop_distractor(df)
    return df[[
        "original_id", "instance", "question",
        "relevant_vars", "relevant_values", "primary_distractor_value",
    ]].rename(columns={"question": "corrupted_prompt"})


def _load_dataset(dataset):
    if dataset in GROUP_DATASETS:
        parts = []
        for part in GROUP_DATASETS[dataset]:
            path = Path(DATA_DIR) / DATASET_CHOICES[part]
            parts.append(pd.read_csv(path))
        df = pd.concat(parts, ignore_index=True)
    else:
        path = Path(DATA_DIR) / DATASET_CHOICES[dataset]
        df = pd.read_csv(path)

    needed = {"relevant_vars", "relevant_values", "primary_distractor_value"}
    if not needed.issubset(df.columns):
        source = _load_noop_source_annotations()
        df = df.merge(
            source,
            on=["original_id", "instance", "corrupted_prompt"],
            how="left",
        )
    return df


def _first_token_id(tokenizer, value):
    if value is None:
        return None
    token_ids = tokenizer.encode(str(value).strip(), add_special_tokens=False)
    if not token_ids:
        return None
    return token_ids[0]


def _single_token_id(tokenizer, value):
    if value is None:
        return None
    token_ids = tokenizer.encode(str(value).strip(), add_special_tokens=False)
    if len(token_ids) != 1:
        return None
    return token_ids[0]


def _coerce_int(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None

def _load_target_spec(row, tokenizer, gold_col="clean_gt_answer", wrong_col="corrupted_gt_answer"):
    relevant_values = _parse_maybe_literal(row.get("relevant_values"))
    if isinstance(relevant_values, dict):
        relevant_items = [
            (var, value)
            for var, raw_value in relevant_values.items()
            if (value := normalize_numeric_value(raw_value)) is not None
        ]
    else:
        relevant_items = []

    gold_value = _coerce_int(row.get(gold_col))
    if gold_value is None:
        return [], [], []

    targets = [
        ("gold_answer", str(gold_value)),
    ]

    wrong_value = _coerce_int(row.get(wrong_col))
    if wrong_value is not None:
        wrong_value = str(wrong_value)
        if wrong_value != targets[0][1]:
            targets.append(("wrong_answer", wrong_value))

    primary_distractor = normalize_numeric_value(row.get("primary_distractor_value"))
    if primary_distractor is not None:
        targets.append(("primary_distractor", primary_distractor))

    for var, value in relevant_items:
        targets.append((f"relevant_{var}", str(value)))

    labels = []
    values = []
    token_ids = []
    for label, value in targets:
        if label == "gold_answer":
            token_id = _first_token_id(tokenizer, value)
        else:
            token_id = _single_token_id(tokenizer, value)
        if token_id is None:
            continue
        labels.append(label)
        values.append(value)
        token_ids.append(token_id)

    return labels, values, token_ids


def _targets_to_series(logits, labels, values):
    result = {}
    for idx, (label, value) in enumerate(zip(labels, values)):
        result[label] = {
            "value": value,
            "logits": to_serializable(logits[:, idx]),
        }
    return result


def _mean_logits(series, prefix):
    rel = [np.array(v["logits"], dtype=float) for k, v in series.items() if k.startswith(prefix)]
    if not rel:
        return None
    return np.nanmean(np.stack(rel, axis=0), axis=0)


def _sanitize_relevant_values(value_map):
    parsed = _parse_maybe_literal(value_map)
    if not isinstance(parsed, dict):
        return None
    cleaned = {
        var: value
        for var, raw_value in parsed.items()
        if (value := normalize_numeric_value(raw_value)) is not None
    }
    return cleaned


def run_noop_target_logit_lens(model, tokenizer, model_name, dataset="noop_correct",
                               verbose=False, translators=None):
    df = _load_dataset(dataset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_suffix = "" if dataset == "direct_all" else f"_{dataset}"
    out_path = OUT_DIR / f"{model_name}_noop_target_logit_lens{dataset_suffix}.jsonl"
    if out_path.exists():
        out_path.unlink()

    is_direct = _dataset_prompt_mode(dataset) == "direct"
    instruction = DIRECT_INSTRUCTION if is_direct else COT_INSTRUCTION

    gold_col, wrong_col = "clean_gt_answer", "corrupted_gt_answer"

    skip_ctr = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=dataset):
        labels, values, token_ids = _load_target_spec(row, tokenizer, gold_col=gold_col, wrong_col=wrong_col)
        if len(token_ids) < 3:
            skip_ctr += 1
            continue

        clean_prompt = row["clean_prompt"]
        corrupted_prompt = row["corrupted_prompt"]

        clean_inputs, _ = tokenize(
            clean_prompt, model, tokenizer,
            instruction=instruction, answer_prefix=ANSWER_PREFIX,
        )
        corrupted_inputs, _ = tokenize(
            corrupted_prompt, model, tokenizer,
            instruction=instruction, answer_prefix=ANSWER_PREFIX,
        )

        clean_state = forward_with_hooks(model, clean_inputs)
        corrupted_state = forward_with_hooks(model, corrupted_inputs)

        clean_logits = get_logit_lens(clean_state, model, token_ids, translators)
        corrupted_logits = get_logit_lens(corrupted_state, model, token_ids, translators)

        clean_series = _targets_to_series(clean_logits, labels, values)
        corrupted_series = _targets_to_series(corrupted_logits, labels, values)

        result = {
            "original_id": int(row["original_id"]),
            "instance": int(row["instance"]),
            "split": int(row["split"]),
            "clean_gt_answer": _coerce_int(row.get("clean_gt_answer")),
            "corrupted_gt_answer": row.get("corrupted_gt_answer"),
            "primary_distractor_value": normalize_numeric_value(row.get("primary_distractor_value")),
            "relevant_values": _sanitize_relevant_values(row.get("relevant_values")),
            "clean_targets": clean_series,
            "corrupted_targets": corrupted_series,
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    if verbose:
        print(f"Skipped {skip_ctr} rows with too few usable target tokens.")
    print(f"Results -> {out_path}")
    return out_path


def plot_noop_target_logit_lens(model_name, dataset="direct_all"):
    dataset_suffix = "" if dataset == "direct_all" else f"_{dataset}"
    results_path = OUT_DIR / f"{model_name}_noop_target_logit_lens{dataset_suffix}.jsonl"
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    def _stack(run_key, target_key):
        arrs = []
        for row in results:
            target = row[run_key].get(target_key)
            if target is not None:
                arrs.append(np.array(target["logits"], dtype=float))
        return np.stack(arrs, axis=0) if arrs else None

    clean_gold = _stack("clean_targets", "gold_answer")
    corrupted_gold = _stack("corrupted_targets", "gold_answer")
    corrupted_wrong = _stack("corrupted_targets", "wrong_answer")

    # Build relevant and distractor over the same subset of rows so panel 2
    # compares apples to apples.
    corrupted_relevant_rows = []
    corrupted_distractor_rows = []
    for row in results:
        n_mean = _mean_logits(row["corrupted_targets"], "relevant_")
        dist = row["corrupted_targets"].get("primary_distractor")
        if n_mean is not None and dist is not None:
            corrupted_relevant_rows.append(n_mean)
            corrupted_distractor_rows.append(np.array(dist["logits"], dtype=float))
    corrupted_relevant = np.stack(corrupted_relevant_rows, axis=0) if corrupted_relevant_rows else None
    corrupted_distractor = np.stack(corrupted_distractor_rows, axis=0) if corrupted_distractor_rows else None

    num_layers = None
    for arr in [clean_gold, corrupted_gold, corrupted_wrong, corrupted_distractor]:
        if arr is not None:
            num_layers = arr.shape[1]
            break
    if num_layers is None:
        raise ValueError(f"No plottable results found in {results_path}")
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {
        "clean": "#2166ac",
        "gold": "#1b7837",
        "wrong": "#d6604d",
        "distractor": "#762a83",
        "relevant": "#b8860b",
    }

    ax = axes[0]
    if corrupted_gold is not None:
        ax.plot(layers, corrupted_gold.mean(axis=0), color=colors["gold"], lw=2, label="NoOp gold")
    if corrupted_wrong is not None:
        ax.plot(layers, corrupted_wrong.mean(axis=0), color=colors["wrong"], lw=2, label="NoOp wrong")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Answer targets")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean logit")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    ax = axes[1]
    if corrupted_relevant is not None:
        ax.plot(layers, corrupted_relevant.mean(axis=0), color=colors["relevant"], lw=2,
                label="NoOp relevant mean")
    if corrupted_relevant is not None and corrupted_distractor is not None:
        ax.plot(layers, corrupted_distractor.mean(axis=0), color=colors["distractor"], lw=2,
                label="NoOp distractor")
    ax.set_title("Value targets")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean logit")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    plt.suptitle(
        f"NoOp target logit lens | {model_name} | N={len(results)} | Dataset={dataset}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()

    save_path = OUT_DIR / f"{model_name}_noop_target_logit_lens{dataset_suffix}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot -> {save_path}")


def _load_results(model_name, dataset):
    dataset_suffix = "" if dataset == "direct_all" else f"_{dataset}"
    results_path = OUT_DIR / f"{model_name}_noop_target_logit_lens{dataset_suffix}.jsonl"
    rows = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def plot_noop_target_panel2_diff(model_name, dataset_a="noop_correct", dataset_b="noop_wrong"):
    def _aggregate(rows):
        rels, dists = [], []
        for row in rows:
            rel = _mean_logits(row["corrupted_targets"], "relevant_")
            dist = row["corrupted_targets"].get("primary_distractor")
            if rel is not None:
                rels.append(rel)
            if dist is not None:
                dists.append(np.array(dist["logits"], dtype=float))
        rel_mean = np.nanmean(np.stack(rels, axis=0), axis=0) if rels else None
        dist_mean = np.nanmean(np.stack(dists, axis=0), axis=0) if dists else None
        return rel_mean, dist_mean, len(rows)

    rows_a = _load_results(model_name, dataset_a)
    rows_b = _load_results(model_name, dataset_b)

    rel_a, dist_a, n_a = _aggregate(rows_a)
    rel_b, dist_b, n_b = _aggregate(rows_b)

    num_layers = next(
        arr.shape[0] for arr in [rel_a, dist_a, rel_b, dist_b] if arr is not None
    )
    layers = np.arange(num_layers)

    fig, ax = plt.subplots(figsize=(8, 5))
    if rel_a is not None and rel_b is not None:
        ax.plot(layers, rel_a - rel_b, color="#b8860b", lw=2, label="relevant mean diff")
    if dist_a is not None and dist_b is not None:
        ax.plot(layers, dist_a - dist_b, color="#762a83", lw=2, label="distractor diff")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(f"Value targets: {dataset_a} − {dataset_b}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean logit diff")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.suptitle(
        f"{model_name} | N {dataset_a}={n_a}, N {dataset_b}={n_b}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    save_path = OUT_DIR / f"{model_name}_noop_target_panel2_diff_{dataset_a}_vs_{dataset_b}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {save_path}")


def plot_noop_target_comparison(model_name, dataset_a, dataset_b):

    def _mean_series(rows, run_key, target_key):
        arrs = []
        for row in rows:
            target = row[run_key].get(target_key)
            if target is not None:
                arrs.append(np.array(target["logits"], dtype=float))
        if not arrs:
            return None
        return np.nanmean(np.stack(arrs, axis=0), axis=0)

    def _mean_relevant_gap(rows):
        gaps = []
        for row in rows:
            rel = _mean_logits(row["corrupted_targets"], "relevant_")
            dist = row["corrupted_targets"].get("primary_distractor")
            if rel is None or dist is None:
                continue
            gaps.append(rel - np.array(dist["logits"], dtype=float))
        if not gaps:
            return None
        return np.nanmean(np.stack(gaps, axis=0), axis=0)

    rows_a = _load_results(model_name, dataset_a)
    rows_b = _load_results(model_name, dataset_b)

    gold_gap_a = _mean_series(rows_a, "corrupted_targets", "gold_answer")
    dist_a = _mean_series(rows_a, "corrupted_targets", "primary_distractor")
    wrong_a = _mean_series(rows_a, "corrupted_targets", "wrong_answer")
    rel_gap_a = _mean_relevant_gap(rows_a)

    gold_gap_b = _mean_series(rows_b, "corrupted_targets", "gold_answer")
    dist_b = _mean_series(rows_b, "corrupted_targets", "primary_distractor")
    wrong_b = _mean_series(rows_b, "corrupted_targets", "wrong_answer")
    rel_gap_b = _mean_relevant_gap(rows_b)

    series_for_len = [gold_gap_a, dist_a, wrong_a, gold_gap_b, dist_b, wrong_b]
    num_layers = next(arr.shape[0] for arr in series_for_len if arr is not None)
    layers = np.arange(num_layers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"a": "#1b7837", "b": "#8c510a"}

    ax = axes[0]
    if gold_gap_a is not None and wrong_a is not None:
        ax.plot(layers, gold_gap_a - wrong_a, color=colors["a"], lw=2, label=f"{dataset_a}: gold-wrong")
    if gold_gap_b is not None and wrong_b is not None:
        ax.plot(layers, gold_gap_b - wrong_b, color=colors["b"], lw=2, label=f"{dataset_b}: gold-wrong")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Gold - wrong answer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean logit gap")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    ax = axes[1]
    if rel_gap_a is not None:
        ax.plot(layers, rel_gap_a, color=colors["a"], lw=2, label=f"{dataset_a}: relevant-dist")
    if rel_gap_b is not None:
        ax.plot(layers, rel_gap_b, color=colors["b"], lw=2, label=f"{dataset_b}: relevant-dist")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Relevant mean - distractor")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean logit gap")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)

    plt.suptitle(
        f"NoOp target comparison | {model_name} | {dataset_a} vs {dataset_b}",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    save_path = OUT_DIR / f"{model_name}_noop_target_compare_{dataset_a}_vs_{dataset_b}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot -> {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct",
                        choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--dataset", type=str,
                        default="direct_all",
                        choices=list(DATASET_CHOICES.keys()) + list(GROUP_DATASETS.keys()))
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--tuned_lens", action="store_true",
                        help="Use tuned lens translators if available")
    parser.add_argument("--compare_to", type=str, default=None,
                        choices=list(DATASET_CHOICES.keys()) + list(GROUP_DATASETS.keys()),
                        help="If set, make a comparison plot between --dataset and this dataset.")
    parser.add_argument("--panel2_diff_b", type=str, default=None,
                        choices=list(DATASET_CHOICES.keys()) + list(GROUP_DATASETS.keys()),
                        help="If set, plot panel-2 diff (--dataset minus this dataset).")
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]
    translators = None
    if args.tuned_lens:
        translators = load_translators(model_name)
        if translators is None:
            print("WARNING: tuned lens weights not found, falling back to standard logit lens")
        else:
            print(f"Loaded {len(translators)} tuned lens translators")

    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        run_noop_target_logit_lens(
            model, tokenizer, model_name,
            dataset=args.dataset, verbose=args.verbose, translators=translators,
        )

    plot_noop_target_logit_lens(model_name, dataset=args.dataset)
    if args.compare_to is not None:
        plot_noop_target_comparison(model_name, args.dataset, args.compare_to)
    if args.panel2_diff_b is not None:
        plot_noop_target_panel2_diff(model_name, args.dataset, args.panel2_diff_b)
