"""
Attention analysis: does the model attend to the distractor tokens in the NoOp run?

For each example in a direct-patching dataset:
  1. Tokenize the corrupted (NoOp) prompt and locate the inserted distractor tokens.
  2. Run one forward pass with output_attentions=True.
  3. At each layer, compare mean attention from the last token to:
       - distractor positions (the inserted sentence)
       - control positions   (the shared prefix, same sequence, same softmax)
  4. excess = distractor − control is the within-sequence signal.

Compares direct_sym_abs_wrong vs direct_sym_abs_correct to test whether excess
distractor attention predicts errors on the symbolic abstraction task.

Usage:
    python run_attention_analysis.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_attention_analysis.py --plot_only
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from config import (
    DATA_DIR, LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP,
    DIRECT_INSTRUCTION, ANSWER_PREFIX,
)
from utils.noop_utils import (
    load_model, DATASET_CHOICES,
    load_noop_source_annotations, normalize_numeric_value,
)

ATTN_RESULT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "attention_analysis"
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Changed-position detection
# ---------------------------------------------------------------------------

def get_changed_positions(sym_ids: list, noop_ids: list):
    """Return (sym_changed, noop_changed): token indices in the changed span.

    Finds the longest shared prefix and longest shared suffix; everything
    between them is treated as the contiguous substituted region. Depending on
    the caller/dataset, this can be just substituted number tokens or an entire
    inserted/replaced clause (e.g. a noop distractor sentence).

    This helper assumes the two token sequences differ in one contiguous region.
    If there are multiple independent edits, the returned span will include
    all tokens between the first and last edit.
    """
    n_pre = 0
    for a, b in zip(sym_ids, noop_ids):
        if a == b:
            n_pre += 1
        else:
            break

    n_suf = 0
    for a, b in zip(reversed(sym_ids), reversed(noop_ids)):
        if n_pre + n_suf + 1 >= min(len(sym_ids), len(noop_ids)):
            break
        if a == b:
            n_suf += 1
        else:
            break

    sym_end  = len(sym_ids)  - n_suf if n_suf > 0 else len(sym_ids)
    noop_end = len(noop_ids) - n_suf if n_suf > 0 else len(noop_ids)
    return list(range(n_pre, sym_end)), list(range(n_pre, noop_end))


# ---------------------------------------------------------------------------
# Per-example attention extraction
# ---------------------------------------------------------------------------

def _run_batch(model, batch_ids, batch_position_sets, pad_token_id):
    """Left-pad a batch of variable-length sequences and run one forward pass.

    batch_ids:          list of B token-id lists (different lengths)
    batch_position_sets: list of B × [positions_a, positions_b, ...] (unpadded indices)

    Returns list of B × ([layer_result, ...], [head_result, ...]) tuples,
    where each layer_result is (n_layers,) and each head_result is (n_layers, n_heads).
    """
    max_len    = max(len(ids) for ids in batch_ids)
    real_lens  = [len(ids) for ids in batch_ids]
    n_sets     = len(batch_position_sets[0])

    # Right-pad: real content at positions 0..real_len-1, same as unpadded → preserves RoPE
    padded    = [ids + [pad_token_id] * (max_len - len(ids)) for ids in batch_ids]
    input_t   = torch.tensor(padded, dtype=torch.long).to(model.device)
    attn_mask = torch.tensor([[1] * len(ids) + [0] * (max_len - len(ids))
                               for ids in batch_ids],
                              dtype=torch.long).to(model.device)

    with torch.no_grad():
        out = model(input_t, attention_mask=attn_mask, output_attentions=True)

    n_layers = len(out.attentions)
    n_heads  = out.attentions[0].shape[1]

    layer_res = [[np.zeros(n_layers)            for _ in range(n_sets)] for _ in batch_ids]
    head_res  = [[np.zeros((n_layers, n_heads)) for _ in range(n_sets)] for _ in batch_ids]

    for i, attn in enumerate(out.attentions):
        attn_cpu = attn.float().cpu()  # (B, n_heads, max_len, max_len)
        for b, (pos_sets, last_pos) in enumerate(zip(batch_position_sets,
                                                      [l - 1 for l in real_lens])):
            last_row = attn_cpu[b, :, last_pos, :]  # (n_heads, max_len) — last real token
            for s, positions in enumerate(pos_sets):
                ha = last_row[:, positions].mean(dim=1)  # (n_heads,) — positions unchanged
                layer_res[b][s][i] = ha.mean().item()
                head_res[b][s][i]  = ha.numpy()

    return [(layer_res[b], head_res[b]) for b in range(len(batch_ids))]


# ---------------------------------------------------------------------------
# Dataset loading and value-position helpers
# ---------------------------------------------------------------------------

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


def _find_value_positions(value_str, full_str, offsets, char_lo, char_hi):
    """Token indices whose char span overlaps any occurrence of value_str in full_str[char_lo:char_hi]."""
    positions = []
    if not value_str:
        return positions

    # Match standalone values only. Plain substring search makes "4" match
    # "14", "40", and "1/4", which contaminates value-level attention.
    pattern = re.compile(
        rf"(?<![A-Za-z0-9./]){re.escape(value_str)}(?![A-Za-z0-9./])",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(full_str, char_lo, char_hi):
        idx = match.start()
        val_end = match.end()
        for i, (s, e) in enumerate(offsets):
            if s < val_end and e > idx and i not in positions:
                positions.append(i)
    return positions


def _load_dataset(ds: str) -> pd.DataFrame:
    """Load dataset CSV, merging relevant_values + primary_distractor_value if absent."""
    df = pd.read_csv(Path(DATA_DIR) / DATASET_CHOICES[ds])
    needed = {"relevant_values", "primary_distractor_value"}
    if not needed.issubset(df.columns):
        source = load_noop_source_annotations()
        if not source.empty:
            df = df.merge(source, on=["original_id", "instance", "corrupted_prompt"],
                          how="left")
    return df


# ---------------------------------------------------------------------------
# Main analysis loop
# ---------------------------------------------------------------------------

def run_attention_analysis(model, tokenizer, model_name: str,
                            datasets: list[str], batch_size: int = 4) -> dict:
    ATTN_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    raw_arrays  = {}  # ds -> raw per-example arrays, kept for combination

    def fmt(q):
        msgs = [{"role": "user", "content": f"{q}\n{DIRECT_INSTRUCTION}"}]
        base = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return base + ANSWER_PREFIX

    # Precompute how many tokens belong to the chat-template instruction wrapper
    # (everything before the actual problem text). Uses a unique probe string and
    # offset mapping to avoid BPE boundary ambiguity.
    _probe_text = "PROBLEM_START_PROBE"
    _probe_full = fmt(_probe_text)
    _probe_char = _probe_full.find(_probe_text)
    _probe_enc  = tokenizer(_probe_full, add_special_tokens=False,
                            return_offsets_mapping=True)
    _instruction_tokens = next(
        i for i, (s, _) in enumerate(_probe_enc["offset_mapping"])
        if s >= _probe_char
    )

    for ds in datasets:
        df = _load_dataset(ds)

        clause_rows        = []
        prefix_rows        = []
        rel_val_rows       = []
        dist_val_rows      = []
        clause_head_rows   = []
        prefix_head_rows   = []
        rel_val_head_rows  = []
        dist_val_head_rows = []
        value_valid        = []
        n_skip             = 0
        pending            = []
        pad_id             = tokenizer.pad_token_id

        def flush(pending):
            if not pending:
                return
            results = _run_batch(
                model,
                [p["ids"]       for p in pending],
                [p["positions"] for p in pending],
                pad_id,
            )
            for (lr, hr), p in zip(results, pending):
                clause_rows.append(lr[0]);       prefix_rows.append(lr[1])
                rel_val_rows.append(lr[2]);      dist_val_rows.append(lr[3])
                clause_head_rows.append(hr[0]);  prefix_head_rows.append(hr[1])
                rel_val_head_rows.append(hr[2]); dist_val_head_rows.append(hr[3])
                value_valid.append(p["has_values"])
            pending.clear()

        for _, row in tqdm(df.iterrows(), total=len(df), desc=ds):
            sym_ids  = tokenizer(fmt(row["clean_prompt"]), add_special_tokens=False)["input_ids"]
            enc_noop = tokenizer(fmt(row["corrupted_prompt"]), add_special_tokens=False,
                                 return_offsets_mapping=True)
            noop_ids = enc_noop["input_ids"]
            offsets  = enc_noop["offset_mapping"]

            sym_changed, noop_changed = get_changed_positions(sym_ids, noop_ids)
            if not noop_changed:
                n_skip += 1
                continue

            n_pre            = noop_changed[0]
            prefix_positions = list(range(_instruction_tokens, n_pre))
            if not prefix_positions:
                n_skip += 1
                continue

            clause_positions = noop_changed

            # Character boundaries for value-position search
            full_noop_str   = fmt(row["corrupted_prompt"])
            problem_char_lo = offsets[_instruction_tokens][0] if _instruction_tokens < len(offsets) else 0
            noop_char_lo    = offsets[noop_changed[0]][0]
            noop_char_hi    = offsets[noop_changed[-1]][1]

            # Relevant value token positions — problem body only (before noop clause)
            rel_val_pos = []
            rel_vals_raw = _parse_maybe_literal(row.get("relevant_values"))
            if isinstance(rel_vals_raw, dict):
                for val in rel_vals_raw.values():
                    val_str = normalize_numeric_value(val)
                    if val_str:
                        rel_val_pos.extend(
                            _find_value_positions(val_str, full_noop_str, offsets,
                                                  problem_char_lo, noop_char_lo)
                        )
            rel_val_pos = sorted(set(rel_val_pos))

            # Distractor value token positions — within noop clause only
            dist_val_pos = []
            dist_str = normalize_numeric_value(row.get("primary_distractor_value"))
            if dist_str:
                dist_val_pos = sorted(set(
                    _find_value_positions(dist_str, full_noop_str, offsets,
                                         noop_char_lo, noop_char_hi)
                ))

            has_values = bool(rel_val_pos) and bool(dist_val_pos)

            # Always pass 4 position sets; fall back so batches stay uniform
            pending.append({
                "ids":       noop_ids,
                "positions": [
                    clause_positions,
                    prefix_positions,
                    rel_val_pos  if rel_val_pos  else prefix_positions,
                    dist_val_pos if dist_val_pos else clause_positions,
                ],
                "has_values": has_values,
            })
            if len(pending) >= batch_size:
                flush(pending)

        flush(pending)

        N = len(clause_rows)
        print(f"  {ds}: N={N}, skipped={n_skip}")

        if N == 0:
            print(f"  {ds}: no valid samples, skipping")
            continue

        clause_arr     = np.array(clause_rows)        # (N, L)
        prefix_arr     = np.array(prefix_rows)        # (N, L)
        rel_val_arr    = np.array(rel_val_rows)       # (N, L)
        dist_val_arr   = np.array(dist_val_rows)      # (N, L)
        clause_h_arr   = np.array(clause_head_rows)   # (N, L, H)
        prefix_h_arr   = np.array(prefix_head_rows)   # (N, L, H)
        rel_val_h_arr  = np.array(rel_val_head_rows)  # (N, L, H)
        dist_val_h_arr = np.array(dist_val_head_rows) # (N, L, H)
        val_mask       = np.array(value_valid, dtype=bool)

        excess_clause      = clause_arr - prefix_arr
        excess_clause_head = clause_h_arr - prefix_h_arr
        N_values = int(val_mask.sum())
        print(f"  {ds}: N_values={N_values}")

        result = {
            "clause_mean":             clause_arr.mean(0).tolist(),
            "prefix_mean":             prefix_arr.mean(0).tolist(),
            "excess_clause_mean":      excess_clause.mean(0).tolist(),
            "clause_p25":              np.percentile(clause_arr,   25, axis=0).tolist(),
            "clause_p75":              np.percentile(clause_arr,   75, axis=0).tolist(),
            "prefix_p25":              np.percentile(prefix_arr,   25, axis=0).tolist(),
            "prefix_p75":              np.percentile(prefix_arr,   75, axis=0).tolist(),
            "excess_clause_p25":       np.percentile(excess_clause, 25, axis=0).tolist(),
            "excess_clause_p75":       np.percentile(excess_clause, 75, axis=0).tolist(),
            "clause_head_mean":        clause_h_arr.mean(0).tolist(),
            "prefix_head_mean":        prefix_h_arr.mean(0).tolist(),
            "excess_clause_head_mean": excess_clause_head.mean(0).tolist(),
            "n":        N,
            "n_values": N_values,
        }

        ev_arr = None
        if N_values > 0:
            ev_arr  = dist_val_arr[val_mask]  - rel_val_arr[val_mask]
            ev_head = dist_val_h_arr[val_mask] - rel_val_h_arr[val_mask]
            result.update({
                "dist_val_mean":           dist_val_arr[val_mask].mean(0).tolist(),
                "rel_val_mean":            rel_val_arr[val_mask].mean(0).tolist(),
                "excess_values_mean":      ev_arr.mean(0).tolist(),
                "excess_values_p25":       np.percentile(ev_arr, 25, axis=0).tolist(),
                "excess_values_p75":       np.percentile(ev_arr, 75, axis=0).tolist(),
                "dist_val_head_mean":      dist_val_h_arr[val_mask].mean(0).tolist(),
                "rel_val_head_mean":       rel_val_h_arr[val_mask].mean(0).tolist(),
                "excess_values_head_mean": ev_head.mean(0).tolist(),
            })

        all_results[ds] = result
        raw_arrays[ds]  = {"excess_clause": excess_clause, "excess_values": ev_arr}

        out_path = ATTN_RESULT_DIR / f"{model_name}_attn_{ds}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved → {out_path}")

        save_kwargs = {"excess_clause": excess_clause}
        if ev_arr is not None:
            save_kwargs["excess_values"] = ev_arr
        raw_path = ATTN_RESULT_DIR / f"{model_name}_attn_{ds}_raw.npz"
        np.savez_compressed(raw_path, **save_kwargs)
        print(f"  Saved → {raw_path}")

    # Level-1 aggregates: leaf JSONs are now on disk, combine them
    _combine_level1(model_name)

    return all_results


# ---------------------------------------------------------------------------
# JSON combination (used after inference and in --plot_only mode)
# ---------------------------------------------------------------------------

def _combine_level1(model_name: str):
    combine_jsons(model_name, "noop_correct",
                  "noop_correct_sym_abs_correct", "noop_correct_sym_abs_wrong")
    combine_jsons(model_name, "noop_wrong",
                  "noop_wrong_sym_abs_correct", "noop_wrong_sym_abs_wrong")


def combine_jsons(model_name: str, out_name: str, key_a: str, key_b: str):
    """Combine two saved JSON result files into one by weighted average.

    Comparison-1 keys are weighted by n; comparison-2 keys by n_values.
    Percentiles are approximated as weighted averages — adequate for plotting.
    """
    pa = ATTN_RESULT_DIR / f"{model_name}_attn_{key_a}.json"
    pb = ATTN_RESULT_DIR / f"{model_name}_attn_{key_b}.json"
    if not pa.exists() or not pb.exists():
        print(f"  combine_jsons: missing {key_a} or {key_b}, skipping {out_name}")
        return
    da = json.loads(pa.read_text())
    db = json.loads(pb.read_text())
    na,  nb  = da["n"],              db["n"]
    nva, nvb = da.get("n_values", 0), db.get("n_values", 0)
    N, NV = na + nb, nva + nvb

    def wavg(key, wa, wb, wt):
        return ((np.array(da[key]) * wa + np.array(db[key]) * wb) / wt).tolist()

    result = {k: wavg(k, na, nb, N) for k in [
        "clause_mean", "prefix_mean", "excess_clause_mean",
        "clause_p25", "clause_p75", "prefix_p25", "prefix_p75",
        "excess_clause_p25", "excess_clause_p75",
        "clause_head_mean", "prefix_head_mean", "excess_clause_head_mean",
    ]}
    result["n"] = N
    result["n_values"] = NV

    val_keys = [
        "dist_val_mean", "rel_val_mean", "excess_values_mean",
        "excess_values_p25", "excess_values_p75",
        "dist_val_head_mean", "rel_val_head_mean", "excess_values_head_mean",
    ]
    if NV > 0:
        for k in val_keys:
            a_has = k in da and nva > 0
            b_has = k in db and nvb > 0
            if a_has and b_has:
                result[k] = wavg(k, nva, nvb, NV)
            elif a_has:
                result[k] = da[k]
            elif b_has:
                result[k] = db[k]

    out_path = ATTN_RESULT_DIR / f"{model_name}_attn_{out_name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"  {out_name}: N={N}, N_values={NV} (combined) → {out_path}")


RAW_COMBINE_MAP = {
    "noop_correct": ("noop_correct_sym_abs_correct", "noop_correct_sym_abs_wrong"),
    "noop_wrong": ("noop_wrong_sym_abs_correct", "noop_wrong_sym_abs_wrong"),
}


def _load_raw_arrays(model_name: str, key: str) -> dict:
    """Return {"clause": ndarray|None, "values": ndarray|None} for bootstrap CIs."""
    raw_path = ATTN_RESULT_DIR / f"{model_name}_attn_{key}_raw.npz"
    if raw_path.exists():
        with np.load(raw_path) as data:
            return {
                "clause": data["excess_clause"] if "excess_clause" in data else None,
                "values": data["excess_values"] if "excess_values" in data else None,
            }

    children = RAW_COMBINE_MAP.get(key)
    if children is None:
        return {"clause": None, "values": None}

    clause_parts, values_parts = [], []
    for child in children:
        child_raw = _load_raw_arrays(model_name, child)
        if child_raw["clause"] is not None:
            clause_parts.append(child_raw["clause"])
        if child_raw["values"] is not None:
            values_parts.append(child_raw["values"])
    return {
        "clause": np.concatenate(clause_parts, axis=0) if clause_parts else None,
        "values": np.concatenate(values_parts, axis=0) if values_parts else None,
    }


def _bootstrap_mean_diff_ci(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    n_boot: int = BOOTSTRAP_SAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, arr_a.shape[0], size=(n_boot, arr_a.shape[0]))
    idx_b = rng.integers(0, arr_b.shape[0], size=(n_boot, arr_b.shape[0]))
    boot_diff = arr_a[idx_a].mean(axis=1) - arr_b[idx_b].mean(axis=1)
    lo = np.percentile(boot_diff, 100 * (alpha / 2), axis=0)
    hi = np.percentile(boot_diff, 100 * (1 - alpha / 2), axis=0)
    return lo, hi


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DATASET_LABELS = {
    "noop_correct_sym_abs_correct": "noop✓ sym✓",
    "noop_correct_sym_abs_wrong":   "noop✓ sym✗",
    "noop_wrong_sym_abs_correct":   "noop✗ sym✓",
    "noop_wrong_sym_abs_wrong":     "noop✗ sym✗",
    "noop_correct":                 "noop correct",
    "noop_wrong":                   "noop wrong",
}

# Color encoding:
#   green  (#1b7837) = noop correct (Level 1)
#   brown  (#8c510a) = noop wrong   (Level 1)
#   blue   (#2166ac) = sym abs correct (Level 2, consistent across panels)
#   red    (#d6604d) = sym abs wrong   (Level 2, consistent across panels)
DATASET_COLORS = {
    "noop_correct":                 "#1b7837",
    "noop_wrong":                   "#8c510a",
    "noop_correct_sym_abs_correct": "#2166ac",
    "noop_correct_sym_abs_wrong":   "#d6604d",
    "noop_wrong_sym_abs_correct":   "#2166ac",
    "noop_wrong_sym_abs_wrong":     "#d6604d",
}


def _heatmap(ax, matrix, title, vmin=None, vmax=None, cmap="RdBu_r"):
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xlabel("Head", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.03)


def plot_three_comparisons(model_name: str):
    """Three figures, one per comparison, each with two rows of panels.

    Row 0 — clause excess  (noop clause vs problem prefix)
    Row 1 — value excess   (distractor value vs relevant values), if available

    Comparisons:
      comparison_1: noop_correct vs noop_wrong
      comparison_2: noop_correct sym✓ vs sym✗
      comparison_3: noop_wrong   sym✓ vs sym✗
    """
    needed = ["noop_correct", "noop_wrong",
              "noop_correct_sym_abs_correct", "noop_correct_sym_abs_wrong",
              "noop_wrong_sym_abs_correct",   "noop_wrong_sym_abs_wrong"]
    data = {}
    for ds in needed:
        p = ATTN_RESULT_DIR / f"{model_name}_attn_{ds}.json"
        if not p.exists():
            print(f"Missing: {p}")
            return
        data[ds] = json.loads(p.read_text())

    L = len(data["noop_correct"]["excess_clause_mean"])
    layers = np.arange(L)

    comparisons = [
        ("noop_correct",                 "noop_wrong",
         "Level 1: noop correct vs wrong",        "comparison_1"),
        ("noop_correct_sym_abs_correct", "noop_correct_sym_abs_wrong",
         "Level 2 (noop correct): sym✓ vs sym✗",  "comparison_2"),
        ("noop_wrong_sym_abs_correct",   "noop_wrong_sym_abs_wrong",
         "Level 2 (noop wrong): sym✓ vs sym✗",    "comparison_3"),
    ]

    for key_a, key_b, title, fname in comparisons:
        ca = DATASET_COLORS[key_a]
        cb = DATASET_COLORS[key_b]
        la = DATASET_LABELS.get(key_a, key_a)
        lb = DATASET_LABELS.get(key_b, key_b)
        da, db = data[key_a], data[key_b]
        raw_a = _load_raw_arrays(model_name, key_a)
        raw_b = _load_raw_arrays(model_name, key_b)

        has_values = ("excess_values_mean" in da and "excess_values_mean" in db)
        nrows = 2 if has_values else 1

        # ── Line plots ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(nrows, 2, figsize=(14, 5 * nrows), squeeze=False)

        # Row 0: clause excess
        ec_a  = np.array(da["excess_clause_mean"])
        ec_b  = np.array(db["excess_clause_mean"])
        ec_lo = ec_hi = None
        if raw_a["clause"] is not None and raw_b["clause"] is not None:
            ec_lo, ec_hi = _bootstrap_mean_diff_ci(raw_a["clause"], raw_b["clause"])

        ax = axes[0, 0]
        ax.plot(layers, ec_a, color=ca, lw=2, label=f"{la} (N={da['n']})")
        ax.fill_between(layers, da["excess_clause_p25"], da["excess_clause_p75"],
                        alpha=0.15, color=ca)
        ax.plot(layers, ec_b, color=cb, lw=2, label=f"{lb} (N={db['n']})")
        ax.fill_between(layers, db["excess_clause_p25"], db["excess_clause_p75"],
                        alpha=0.15, color=cb)
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title("Clause excess  (clause − prefix)", fontsize=12)
        ax.set_ylabel("Mean attention", fontsize=10)
        ax.legend(fontsize=10)

        ax = axes[0, 1]
        ax.plot(layers, ec_a - ec_b, color="#1a1a1a", lw=2)
        if ec_lo is not None:
            ax.fill_between(layers, ec_lo, ec_hi, alpha=0.15, color="#1a1a1a")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"Clause diff: {la} − {lb}", fontsize=12)
        ax.set_ylabel("Attention diff", fontsize=10)

        # Row 1: value excess
        if has_values:
            ev_a  = np.array(da["excess_values_mean"])
            ev_b  = np.array(db["excess_values_mean"])
            ev_lo = ev_hi = None
            if raw_a["values"] is not None and raw_b["values"] is not None:
                ev_lo, ev_hi = _bootstrap_mean_diff_ci(raw_a["values"], raw_b["values"])

            ax = axes[1, 0]
            ax.plot(layers, ev_a, color=ca, lw=2,
                    label=f"{la} (N={da.get('n_values', '?')})")
            ax.fill_between(layers, da["excess_values_p25"], da["excess_values_p75"],
                            alpha=0.15, color=ca)
            ax.plot(layers, ev_b, color=cb, lw=2,
                    label=f"{lb} (N={db.get('n_values', '?')})")
            ax.fill_between(layers, db["excess_values_p25"], db["excess_values_p75"],
                            alpha=0.15, color=cb)
            ax.axhline(0, color="black", lw=0.8, alpha=0.5)
            ax.set_title("Value excess  (distractor val − relevant vals)", fontsize=12)
            ax.set_ylabel("Mean attention", fontsize=10)
            ax.legend(fontsize=10)

            ax = axes[1, 1]
            ax.plot(layers, ev_a - ev_b, color="#1a1a1a", lw=2)
            if ev_lo is not None:
                ax.fill_between(layers, ev_lo, ev_hi, alpha=0.15, color="#1a1a1a")
            ax.axhline(0, color="black", lw=0.8, alpha=0.5)
            ax.set_title(f"Value diff: {la} − {lb}", fontsize=12)
            ax.set_ylabel("Attention diff", fontsize=10)

        for ax in axes.flat:
            ax.set_xlabel("Layer", fontsize=11)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_xticks(layers[::5])

        plt.suptitle(f"{title}  |  {model_name}", fontsize=13, y=1.02)
        plt.tight_layout()
        save_path = ATTN_RESULT_DIR / f"{model_name}_{fname}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot → {save_path}")

        # ── Heatmaps ──────────────────────────────────────────────────────────
        has_val_heads = (
            "excess_values_head_mean" in da and "excess_values_head_mean" in db
        )
        hm_rows = 2 if has_val_heads else 1
        fig, axes = plt.subplots(hm_rows, 3, figsize=(21, 8 * hm_rows), squeeze=False)

        ha    = np.array(da["excess_clause_head_mean"])
        hb    = np.array(db["excess_clause_head_mean"])
        hdiff = ha - hb
        elim  = max(abs(ha).max(), abs(hb).max())
        dlim  = abs(hdiff).max()
        _heatmap(axes[0, 0], ha,    f"{la} (N={da['n']})\nclause excess", vmin=-elim, vmax=elim)
        _heatmap(axes[0, 1], hb,    f"{lb} (N={db['n']})\nclause excess", vmin=-elim, vmax=elim)
        _heatmap(axes[0, 2], hdiff, f"Clause diff: {la} − {lb}",          vmin=-dlim, vmax=dlim)

        if has_val_heads:
            va    = np.array(da["excess_values_head_mean"])
            vb    = np.array(db["excess_values_head_mean"])
            vdiff = va - vb
            vlim  = max(abs(va).max(), abs(vb).max())
            vdlim = abs(vdiff).max()
            _heatmap(axes[1, 0], va,    f"{la} (N={da.get('n_values','?')})\nvalue excess", vmin=-vlim, vmax=vlim)
            _heatmap(axes[1, 1], vb,    f"{lb} (N={db.get('n_values','?')})\nvalue excess", vmin=-vlim, vmax=vlim)
            _heatmap(axes[1, 2], vdiff, f"Value diff: {la} − {lb}",                         vmin=-vdlim, vmax=vdlim)

        plt.suptitle(f"{title} — per head  |  {model_name}", fontsize=13, y=1.01)
        plt.tight_layout()
        hm_path = ATTN_RESULT_DIR / f"{model_name}_{fname}_head.png"
        plt.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot → {hm_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct",
                        choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--datasets", nargs="+",
                        default=["noop_correct_sym_abs_correct", "noop_correct_sym_abs_wrong",
                                 "noop_wrong_sym_abs_correct",   "noop_wrong_sym_abs_wrong"])
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]

    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        model.set_attn_implementation("eager")
        run_attention_analysis(model, tokenizer, model_name,
                               datasets=args.datasets, batch_size=args.batch_size)
    else:
        _combine_level1(model_name)

    plot_three_comparisons(model_name)
