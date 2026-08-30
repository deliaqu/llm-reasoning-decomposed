"""
Per-head Direct Logit Attribution (DLA) from the distractor value token(s).

At a given layer, for each attention head h:

    attn_h   = Σ_{t ∈ noop_clause_positions} attn_weight_h[answer → t]
    ov_h     = W_U[target] · W_O_h · V_h[noop_clause_pos]   (raw logit for target token)
    dla_h    = attn_h × ov_h

Datasets:
    noop_clean  (default):
        noop_correct  — noop-clean runs the model got right; target = clean_gt_answer
        noop_wrong    — noop-clean runs the model got wrong; targets = corrupted_gt_answer + clean_gt_answer

    fvn_tfm:
        fvn_filler    — filler-question runs (correct); target = answer
        fvn_noop      — noop-question runs (wrong);    targets = noop_model_answer + answer

Attention routing (attn_h) always uses the changed positions between the noop
and clean/filler prompt — that locates the noop/filler clause in the sequence.
For noop_clean, positions are further filtered to the primary_distractor_value token.

Usage:
    python run_noop_dla.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_noop_dla.py --model_id meta-llama/Llama-3.3-70B-Instruct --dataset fvn_tfm
    python run_noop_dla.py --plot_only
    python run_noop_dla.py --plot_only --dataset fvn_tfm
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from config import (
    ANSWER_PREFIX, DIRECT_INSTRUCTION,
    LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP,
)
from utils.noop_utils import load_model, normalize_numeric_value, annotate_primary_noop_distractor
from run_attention_analysis import (
    _find_value_positions, get_changed_positions,
)
from run_noop_target_logit_lens import (
    _coerce_int, _first_token_id, _single_token_id,
)
from run_noop_activation_patching import load_noop_clean_direct

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "dla_analysis"


def layer_dir(model_name, dataset, layer):
    """Path layout matches sibling experiments: {model}/{dataset}/l{layer}/."""
    return OUT_DIR / model_name / dataset / f"l{layer}"


def _leaf_prefix(group):
    """Drop the dataset prefix from a group name when used as a leaf filename;
    dataset context is already encoded in the directory path."""
    if group.startswith("fvn_") or group.startswith("fdf_"):
        return group[4:]
    return group

# Inverted map: file-safe model name → HuggingFace model ID
MODEL_ID_MAP = {v: k for k, v in MODEL_NAME_MAP.items()}

# ---------------------------------------------------------------------------
# noop_clean dataset config
# ---------------------------------------------------------------------------

# Column names after _load_noop_clean_groups() renames noop_clean CSV columns
# to match what _prepare_noop_clean_row and the target specs expect.
GROUP_TARGET_SPECS = {
    "noop_correct": [
        ("clean_answer", "clean_gt_answer", "gold (correct answer)"),
    ],
    "noop_wrong": [
        ("wrong_answer", "corrupted_gt_answer", "dist (model's wrong answer)"),
        ("clean_answer", "clean_gt_answer", "gold (answer it should output)"),
    ],
}


def _load_noop_clean_groups(model_id):
    """Load gsm_noop_clean CSVs, rename columns to match DLA row helper expectations,
    annotate primary_distractor_value, and split into correct/wrong DataFrames."""
    import pandas as pd
    df = load_noop_clean_direct(model_id)
    df = df.rename(columns={
        "question":               "corrupted_prompt",
        "original_question":      "clean_prompt",
        "answer":                 "clean_gt_answer",
        "original_direct_answer": "corrupted_gt_answer",
    })
    df = annotate_primary_noop_distractor(df)
    correct = df[df["original_direct_correctness"] == True].copy()
    wrong   = df[df["original_direct_correctness"] == False].copy()
    return {"noop_correct": correct, "noop_wrong": wrong}

# ---------------------------------------------------------------------------
# fvn_tfm dataset config
# ---------------------------------------------------------------------------

FVN_GROUP_TARGET_SPECS = {
    "fvn_filler": [
        ("clean_answer", "answer", "gold (correct answer)"),
    ],
    "fvn_noop": [
        ("wrong_answer", "noop_model_answer", "dist (model's wrong answer)"),
        ("clean_answer", "answer", "gold (answer it should output)"),
    ],
}

# fdf_tfm shares the same column layout as fvn_tfm; only the CSV path and
# group prefix differ.
FDF_GROUP_TARGET_SPECS = {
    "fdf_filler": [
        ("clean_answer", "answer", "gold (correct answer)"),
    ],
    "fdf_noop": [
        ("wrong_answer", "noop_model_answer", "dist (model's wrong answer)"),
        ("clean_answer", "answer", "gold (answer it should output)"),
    ],
}

GROUP_COLORS = {
    "noop_correct": "#1b7837",
    "noop_wrong":   "#8c510a",
    "fvn_filler":   "#2166ac",
    "fvn_noop":     "#d6604d",
    "fdf_filler":   "#2166ac",
    "fdf_noop":     "#d6604d",
}


def _load_fvn_pairs(model_name):
    import pandas as pd
    model_id = MODEL_ID_MAP.get(model_name, "")
    model_id_short = model_id.split("/")[-1]
    path = (Path(LOGIT_LENS_RESULT_DIR).parent / "disentangled_evaluation" /
            f"gsm_filler_vs_noop_tfm_direct_pairs_filler_correct_noop_wrong_{model_id_short}.csv")
    if not path.exists():
        raise FileNotFoundError(f"fvn_tfm pairs CSV not found: {path}")
    return pd.read_csv(path)


def _load_fdf_pairs(model_name):
    import pandas as pd
    model_id = MODEL_ID_MAP.get(model_name, "")
    model_id_short = model_id.split("/")[-1]
    path = (Path(LOGIT_LENS_RESULT_DIR).parent / "disentangled_evaluation" /
            f"gsm_filler_df_vs_noop_clean_tfm_direct_pairs_filler_correct_noop_wrong_{model_id_short}.csv")
    if not path.exists():
        raise FileNotFoundError(f"fdf_tfm pairs CSV not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Per-row data preparation helpers
# ---------------------------------------------------------------------------

def _prepare_noop_clean_row(row, tokenizer, fmt):
    """Return (forward_ids, dist_positions) for a noop_clean row, or None to skip."""
    pos_str = normalize_numeric_value(row.get("primary_distractor_value"))
    pos_id  = _single_token_id(tokenizer, pos_str) if pos_str else None
    if pos_id is None:
        return None

    enc = tokenizer(fmt(row["corrupted_prompt"]), add_special_tokens=False,
                    return_offsets_mapping=True)
    noop_ids = enc["input_ids"]
    offsets  = enc["offset_mapping"]
    sym_ids  = tokenizer(fmt(row["clean_prompt"]), add_special_tokens=False)["input_ids"]

    _, noop_changed = get_changed_positions(sym_ids, noop_ids)
    if not noop_changed:
        return None

    full_noop_str = fmt(row["corrupted_prompt"])
    noop_char_lo  = offsets[noop_changed[0]][0]
    noop_char_hi  = offsets[noop_changed[-1]][1]

    dist_positions = _find_value_positions(
        pos_str, full_noop_str, offsets, noop_char_lo, noop_char_hi)
    if not dist_positions:
        return None

    return noop_ids, dist_positions


def _prepare_fvn_row(row, group, tokenizer, fmt):
    """Return (forward_ids, dist_positions) for a fvn_tfm/fdf_tfm row.

    Both datasets share the same column layout. Group suffix selects which
    question is forwarded and which set of changed positions is used as `dist`.
    {fvn,fdf}_noop   — forward pass on the noop question;   dist = noop-changed positions.
    {fvn,fdf}_filler — forward pass on the filler question; dist = filler-changed positions.
    """
    noop_str   = fmt(row["noop_question"])
    filler_str = fmt(row["filler_question"])

    noop_ids   = tokenizer(noop_str,   add_special_tokens=False)["input_ids"]
    filler_ids = tokenizer(filler_str, add_special_tokens=False)["input_ids"]

    filler_changed, noop_changed = get_changed_positions(filler_ids, noop_ids)

    if group.endswith("_noop"):
        forward_ids    = noop_ids
        dist_positions = noop_changed
    else:  # *_filler
        forward_ids    = filler_ids
        dist_positions = filler_changed

    if not dist_positions:
        return None

    return forward_ids, dist_positions


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def run_dla(model, tokenizer, model_name,
            groups=None, layer=39, dataset="noop_clean"):
    """`layer` is the paper-display layer index L{layer} (1-indexed: residual
    after `layer` blocks have processed). Internally we index
    model.model.layers[layer - 1]; on disk the leaf dir is `l{layer}/`."""
    if groups is None:
        if dataset == "noop_clean":
            groups = ("noop_correct", "noop_wrong")
        elif dataset == "fvn_tfm":
            groups = ("fvn_filler", "fvn_noop")
        elif dataset == "fdf_tfm":
            groups = ("fdf_filler", "fdf_noop")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = lambda: layer_dir(model_name, dataset, layer)

    cfg        = model.config
    n_heads    = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim   = cfg.hidden_size // n_heads
    kv_groups  = n_heads // n_kv_heads
    d_model    = cfg.hidden_size

    block_idx = layer - 1
    attn_mod = model.model.layers[block_idx].self_attn

    model_dtype  = attn_mod.o_proj.weight.dtype
    W_O_cpu      = attn_mod.o_proj.weight.detach().float().cpu()
    W_O_per_head = W_O_cpu.view(d_model, n_heads, head_dim).permute(1, 0, 2).contiguous()

    def fmt(q):
        msgs = [{"role": "user", "content": f"{q}\n{DIRECT_INSTRUCTION}"}]
        base = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        return base + ANSWER_PREFIX

    captured = {}

    def _hook_vproj(_module, _inp, out):
        captured["v_states"] = out.detach()

    hook_vproj = attn_mod.v_proj.register_forward_hook(_hook_vproj)

    all_results = {}

    # Load noop_clean data once (split into groups) before the group loop.
    _noop_clean_groups = (
        _load_noop_clean_groups(MODEL_ID_MAP[model_name])
        if dataset == "noop_clean" else None
    )

    try:
        for group in groups:
            if dataset == "fvn_tfm":
                df = _load_fvn_pairs(model_name)
                target_specs = FVN_GROUP_TARGET_SPECS[group]
            elif dataset == "fdf_tfm":
                df = _load_fdf_pairs(model_name)
                target_specs = FDF_GROUP_TARGET_SPECS[group]
            else:  # noop_clean
                df = _noop_clean_groups[group]
                target_specs = GROUP_TARGET_SPECS[group]

            attn_rows = []
            target_ov_rows  = {name: [] for name, _, _ in target_specs}
            target_dla_rows = {name: [] for name, _, _ in target_specs}
            n_skip = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc=group):
                # ── target token IDs ─────────────────────────────────────────
                target_ids = {}
                for target_name, target_col, _ in target_specs:
                    target_val = _coerce_int(row.get(target_col))
                    target_id  = _first_token_id(tokenizer, target_val)
                    if target_id is not None:
                        target_ids[target_name] = target_id
                if len(target_ids) != len(target_specs):
                    n_skip += 1; continue

                # ── forward prompt and dist positions ────────────────────────
                if dataset in ("fvn_tfm", "fdf_tfm"):
                    prep = _prepare_fvn_row(row, group, tokenizer, fmt)
                else:
                    prep = _prepare_noop_clean_row(row, tokenizer, fmt)
                if prep is None:
                    n_skip += 1; continue
                forward_ids, dist_positions = prep

                last_pos = len(forward_ids) - 1

                # ── forward pass ─────────────────────────────────────────────
                input_t = torch.tensor([forward_ids]).to(model.device)
                with torch.no_grad():
                    out = model(input_t, output_attentions=True)

                attn_w = out.attentions[block_idx][0].float()   # (n_heads, seq, seq)

                # ── V at dist positions ───────────────────────────────────────
                v_raw     = captured["v_states"][0]
                v_at_dist = v_raw[dist_positions].float().cpu()
                v_kv      = v_at_dist.view(len(dist_positions), n_kv_heads, head_dim)
                v_all     = v_kv.repeat_interleave(kv_groups, dim=1)  # (P, n_heads, head_dim)

                # ── OV score ──────────────────────────────────────────────────
                P       = len(dist_positions)
                M_all   = torch.einsum("hkd,phd->phk", W_O_per_head, v_all)
                M_batch = M_all.reshape(P * n_heads, d_model)
                with torch.no_grad():
                    logits_batch = model.lm_head(
                        M_batch.to(device=attn_w.device, dtype=model_dtype)
                    )

                # ── attention weight: last token → dist positions ─────────────
                attn_to_dist_pos = attn_w[:, last_pos, :][:, dist_positions].transpose(0, 1).cpu()
                attn_to_dist = attn_to_dist_pos.sum(dim=0)  # (n_heads,)

                attn_rows.append(attn_to_dist.numpy())
                for target_name, _, _ in target_specs:
                    ov_per_pos = logits_batch[:, target_ids[target_name]].cpu().view(P, n_heads)
                    dla_all = (attn_to_dist_pos * ov_per_pos).sum(dim=0)
                    ov_all  = torch.where(
                        attn_to_dist.abs() > 1e-12,
                        dla_all / attn_to_dist,
                        ov_per_pos.mean(dim=0),
                    )
                    target_ov_rows[target_name].append(ov_all.numpy())
                    target_dla_rows[target_name].append(dla_all.numpy())

                del out; torch.cuda.empty_cache()

            N = len(attn_rows)
            print(f"  {group}: N={N}, skipped={n_skip}")
            if N == 0:
                print(f"  {group}: no valid samples, skipping")
                continue

            attn_arr = np.array(attn_rows)
            target_results = {}
            raw_arrays = {"attn": attn_arr}
            for target_name, target_col, target_label in target_specs:
                ov_arr  = np.array(target_ov_rows[target_name])
                dla_arr = np.array(target_dla_rows[target_name])
                target_results[target_name] = {
                    "target_col":   target_col,
                    "target_label": target_label,
                    "ov_mean":  ov_arr.mean(0).tolist(),
                    "dla_mean": dla_arr.mean(0).tolist(),
                }
                raw_arrays[f"ov_{target_name}"]  = ov_arr
                raw_arrays[f"dla_{target_name}"] = dla_arr

            primary_target_name = target_specs[0][0]

            result = {
                "attn_mean":    attn_arr.mean(0).tolist(),
                "ov_mean":      target_results[primary_target_name]["ov_mean"],
                "dla_mean":     target_results[primary_target_name]["dla_mean"],
                "target_col":   target_results[primary_target_name]["target_col"],
                "target_label": target_results[primary_target_name]["target_label"],
                "primary_target": primary_target_name,
                "targets":      target_results,
                "n":            N,
            }
            all_results[group] = result

            out_dir().mkdir(parents=True, exist_ok=True)
            leaf = _leaf_prefix(group)
            json_path = out_dir() / f"{leaf}.json"
            json_path.write_text(json.dumps(result, indent=2))
            raw_path  = out_dir() / f"{leaf}_raw.npz"
            np.savez_compressed(raw_path, **raw_arrays)
            print(f"  Saved → {json_path}")

    finally:
        hook_vproj.remove()

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _get_target_block(data, target_name=None):
    targets = data.get("targets")
    if targets:
        name = target_name or data.get("primary_target") or next(iter(targets))
        block = targets[name].copy()
        block["name"] = name
        return block
    return {
        "name": target_name or data.get("primary_target", "primary"),
        "target_col":   data.get("target_col"),
        "target_label": data.get("target_label", "target"),
        "ov_mean":  data["ov_mean"],
        "dla_mean": data["dla_mean"],
    }


def _metrics(data, target_name=None):
    tgt = _get_target_block(data, target_name)["target_label"]
    return [
        ("attn_mean", "Attention weight  (answer → distractor clause position)"),
        ("ov_mean",   f"OV  raw logit for {tgt}"),
        ("dla_mean",  f"DLA = attn × OV  [{tgt}]"),
    ]


def _plot_single(model_name, dataset, group, data, layer, target_name=None):
    col   = GROUP_COLORS.get(group, "#555555")
    n     = data["n"]
    block = _get_target_block(data, target_name)
    plot_data = {
        "attn_mean": data["attn_mean"],
        "ov_mean":   block["ov_mean"],
        "dla_mean":  block["dla_mean"],
    }
    heads = np.arange(len(plot_data["dla_mean"]))
    mets  = _metrics(data, target_name)

    _, axes = plt.subplots(len(mets), 1, figsize=(14, 4 * len(mets)))
    for ax, (key, label) in zip(axes, mets):
        vals = np.array(plot_data[key])
        ax.bar(heads, vals, color=col, alpha=0.85)
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"{label}  |  {group} / {block['name']} (N={n})", fontsize=11)
        ax.set_xlabel("Head")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(f"Per-head DLA at layer {layer}  |  {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()
    suffix = "" if target_name is None else f"_{target_name}"
    out = layer_dir(model_name, dataset, layer)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{_leaf_prefix(group)}{suffix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _plot_comparison(model_name, dataset, data, layer,
                     correct_group="noop_correct", wrong_group="noop_wrong"):
    """
    Comparison layout:
      Row 0 — attn: both groups side-by-side + diff  (directly comparable)
      Row 1 — primary OV:   correct group | wrong group
      Row 2 — primary DLA:  same split as row 1
      Row 3 — clean-answer OV + clean-answer DLA diff  (directly comparable)
    """
    dc, dw = data[correct_group], data[wrong_group]
    nc, nw = dc["n"], dw["n"]
    heads  = np.arange(len(dc["dla_mean"]))
    c_col  = GROUP_COLORS.get(correct_group, "#1b7837")
    w_col  = GROUP_COLORS.get(wrong_group,   "#8c510a")
    c_primary = _get_target_block(dc)
    w_primary = _get_target_block(dw)
    c_clean   = _get_target_block(dc, "clean_answer")
    w_clean   = _get_target_block(dw, "clean_answer")

    fig, axes = plt.subplots(4, 2, figsize=(16, 16))

    # ── Row 0: attention ──────────────────────────────────────────────────────
    attn_c    = np.array(dc["attn_mean"])
    attn_w    = np.array(dw["attn_mean"])
    attn_diff = attn_c - attn_w

    ax = axes[0, 0]
    ax.bar(heads - 0.2, attn_c, width=0.4, color=c_col, alpha=0.8,
           label=f"{correct_group} (N={nc})")
    ax.bar(heads + 0.2, attn_w, width=0.4, color=w_col, alpha=0.8,
           label=f"{wrong_group} (N={nw})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Attention weight (answer → distractor clause)", fontsize=11)
    ax.set_xlabel("Head"); ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[0, 1]
    bar_colors = ["#2166ac" if d > 0 else "#d6604d" for d in attn_diff]
    ax.bar(heads, attn_diff, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    for h in np.argsort(np.abs(attn_diff))[-5:]:
        ax.annotate(str(h), (h, attn_diff[h]),
                    textcoords="offset points",
                    xytext=(0, 4 if attn_diff[h] >= 0 else -10),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_title(f"Attn diff: {correct_group} − {wrong_group}", fontsize=11)
    ax.set_xlabel("Head")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Rows 1-2: primary OV and DLA ──────────────────────────────────────────
    for row_idx, key in enumerate(("ov_mean", "dla_mean"), start=1):
        vc = np.array(c_primary[key])
        vw = np.array(w_primary[key])
        key_lbl = "OV" if key == "ov_mean" else "DLA"

        ax = axes[row_idx, 0]
        ax.bar(heads - 0.2, vc, width=0.4, color=c_col, alpha=0.8,
               label=f"correct — {c_primary['target_label']} (N={nc})")
        ax.bar(heads + 0.2, vw, width=0.4, color=w_col, alpha=0.8,
               label=f"wrong — {w_primary['target_label']} (N={nw})")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(f"{key_lbl} raw logit  [different target tokens — not diff'd]",
                     fontsize=11)
        ax.set_xlabel("Head"); ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        ax = axes[row_idx, 1]
        ax.bar(heads - 0.2, vc, width=0.4, color=c_col, alpha=0.8,
               label=f"correct ({c_primary['target_label']})")
        ax.bar(heads + 0.2, vw, width=0.4, color=w_col, alpha=0.8,
               label=f"wrong ({w_primary['target_label']})")
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        combined_mag = np.abs(vc) + np.abs(vw)
        for h in np.argsort(combined_mag)[-8:]:
            ymax = max(abs(vc[h]), abs(vw[h]))
            ax.annotate(str(h), (h, ymax),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8, fontweight="bold")
        ax.set_title(f"{key_lbl} — top active heads labelled", fontsize=11)
        ax.set_xlabel("Head"); ax.legend(fontsize=8)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    # ── Row 3: clean-answer comparison ────────────────────────────────────────
    clean_ov_c   = np.array(c_clean["ov_mean"])
    clean_ov_w   = np.array(w_clean["ov_mean"])
    clean_dla_c  = np.array(c_clean["dla_mean"])
    clean_dla_w  = np.array(w_clean["dla_mean"])
    clean_dla_diff = clean_dla_c - clean_dla_w

    ax = axes[3, 0]
    ax.bar(heads - 0.2, clean_ov_c, width=0.4, color=c_col, alpha=0.8,
           label=f"{correct_group} clean-answer OV (N={nc})")
    ax.bar(heads + 0.2, clean_ov_w, width=0.4, color=w_col, alpha=0.8,
           label=f"{wrong_group} clean-answer OV (N={nw})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Clean-answer OV across groups", fontsize=11)
    ax.set_xlabel("Head"); ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[3, 1]
    bar_colors = ["#2166ac" if d > 0 else "#d6604d" for d in clean_dla_diff]
    ax.bar(heads, clean_dla_diff, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    for h in np.argsort(np.abs(clean_dla_diff))[-8:]:
        ax.annotate(str(h), (h, clean_dla_diff[h]),
                    textcoords="offset points",
                    xytext=(0, 4 if clean_dla_diff[h] >= 0 else -10),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_title(f"Clean-answer DLA diff: {correct_group} − {wrong_group}", fontsize=11)
    ax.set_xlabel("Head")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(f"Per-head DLA at layer {layer}  |  {model_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    out = layer_dir(model_name, dataset, layer)
    out.mkdir(parents=True, exist_ok=True)
    comparison_path = out / "comparison.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {comparison_path}")

    pos_clean_dla_heads = [h for h in np.argsort(clean_dla_diff)[::-1]
                           if clean_dla_diff[h] > 0][:10]
    instrumental = {
        "comparison": f"{correct_group} clean_answer minus {wrong_group} clean_answer",
        "layer":      layer,
        "model_name": model_name,
        "top_clean_answer_dla_diff_heads": [
            {
                "head":                      int(h),
                "dla_diff":                  float(clean_dla_diff[h]),
                f"{correct_group}_clean_dla": float(clean_dla_c[h]),
                f"{wrong_group}_clean_dla":   float(clean_dla_w[h]),
                f"{correct_group}_clean_ov":  float(clean_ov_c[h]),
                f"{wrong_group}_clean_ov":    float(clean_ov_w[h]),
                "attn_diff":                 float(attn_diff[h]),
            }
            for h in pos_clean_dla_heads
        ],
    }
    summary_path = out / "instrumental_heads.json"
    summary_path.write_text(json.dumps(instrumental, indent=2))
    print(f"Summary → {summary_path}")

    # Scatter: attn vs OV, coloured by DLA
    _, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (group, gdata, tgt_lbl, col) in zip(axes2, [
        (correct_group, dc, c_primary["target_label"], c_col),
        (wrong_group,   dw, w_primary["target_label"], w_col),
    ]):
        block = _get_target_block(gdata)
        attn = np.array(gdata["attn_mean"])
        ov   = np.array(block["ov_mean"])
        dla  = np.array(block["dla_mean"])
        lim  = max(abs(dla).max(), 1e-9)
        sc = ax.scatter(attn, ov, c=dla, cmap="RdBu", vmin=-lim, vmax=lim, s=60, zorder=3)
        plt.colorbar(sc, ax=ax, label=f"DLA [{tgt_lbl}]")
        for h in np.argsort(np.abs(dla))[-10:]:
            ax.annotate(str(h), (attn[h], ov[h]),
                        textcoords="offset points", xytext=(4, 3), fontsize=8)
        ax.set_xlabel("Attn weight (answer → distractor clause)", fontsize=10)
        ax.set_ylabel(f"OV raw logit  [{tgt_lbl}]", fontsize=10)
        ax.set_title(f"{group}  (N={gdata['n']})", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.suptitle(f"Routing vs writing  |  layer {layer}  |  {model_name}", fontsize=12)
    plt.tight_layout()
    scatter_path = out / "scatter.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {scatter_path}")


def plot_dla(model_name, layer=39, dataset="noop_clean"):
    if dataset == "fvn_tfm":
        groups        = ("fvn_filler", "fvn_noop")
        correct_group = "fvn_filler"
        wrong_group   = "fvn_noop"
    elif dataset == "fdf_tfm":
        groups        = ("fdf_filler", "fdf_noop")
        correct_group = "fdf_filler"
        wrong_group   = "fdf_noop"
    else:  # noop_clean
        groups        = ("noop_correct", "noop_wrong")
        correct_group = "noop_correct"
        wrong_group   = "noop_wrong"

    data = {}
    for group in groups:
        p = layer_dir(model_name, dataset, layer) / f"{_leaf_prefix(group)}.json"
        if p.exists():
            data[group] = json.load(open(p))
        else:
            print(f"  (skipping {group} — results not yet available)")

    for group, gdata in data.items():
        targets = gdata.get("targets")
        if targets:
            for target_name in targets:
                _plot_single(model_name, dataset, group, gdata, layer, target_name=target_name)
        else:
            _plot_single(model_name, dataset, group, gdata, layer)

    if len(data) == 2:
        _plot_comparison(model_name, dataset, data, layer,
                         correct_group=correct_group, wrong_group=wrong_group)
    else:
        print("  Comparison plots deferred until both groups are complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct",
                        choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--dataset", choices=["noop_clean", "fvn_tfm", "fdf_tfm"], default="noop_clean",
                        help="Which pair dataset to run DLA on.")
    parser.add_argument("--group",
                        help="Single group to run inference for. "
                             "Omit (with --plot_only) to plot all available results.")
    parser.add_argument(
        "--layer", type=int, default=40,
        help="Paper-display layer L{layer} (1-indexed: residual after `layer` "
             "blocks have processed). Internally maps to model.model.layers[layer-1].",
    )
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip inference; plot whatever results exist.")
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]

    if args.dataset == "noop_clean":
        valid_groups = ("noop_correct", "noop_wrong")
    elif args.dataset == "fvn_tfm":
        valid_groups = ("fvn_filler", "fvn_noop")
    elif args.dataset == "fdf_tfm":
        valid_groups = ("fdf_filler", "fdf_noop")
    else:
        parser.error(f"Unknown dataset: {args.dataset}")
    if args.group and args.group not in valid_groups:
        parser.error(f"--group must be one of {valid_groups} for --dataset {args.dataset}")

    if not args.plot_only:
        if args.group is None:
            parser.error("--group is required for inference (omit only with --plot_only)")
        model, tokenizer = load_model(args.model_id)
        model.set_attn_implementation("eager")
        run_dla(model, tokenizer, model_name,
                groups=[args.group], layer=args.layer, dataset=args.dataset)

    plot_dla(model_name, layer=args.layer, dataset=args.dataset)
