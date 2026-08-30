"""
Per-head Direct Logit Attribution (DLA) on cot_swap pairs, targeted at the
#### token at the readout position (end of shared donor CoT prefix).

Mirrors run_noop_dla but uses cot_swap pair data and the #### readout used by
the cot_swap patching experiments (so the result is directly comparable to
Figure 6 from the paper).

For each pair (filler-side question, noop-side question) with shared
source_cot_prefix appended:

    attn_h   = Σ_{t ∈ clause_positions} attn_weight_h[answer_pos → t]
    ov_h     = W_U[####] · W_O_h · V_h[clause_pos]
    dla_h    = attn_h × ov_h

Run separately for each side: 'filler' on the filler-question prompt with its
own clause positions, 'noop' on the noop-question prompt with its own clause
positions. Both sides share the same donor CoT prefix, so the common suffix
spans the CoT and everything after — get_changed_positions correctly returns
only the inserted clause tokens even when the two sides have different lengths.

Multi-layer in a single forward pass: hook v_proj on every requested layer
and compute per-layer DLA from the captured V states + the layer's attention
weights, so a sweep over L22..L36 costs ~|rows| forwards, not |rows| × |layers|.

Usage:
    python run_cot_swap_dla.py --layers 22 23 24 ... 36
    python run_cot_swap_dla.py --plot_only --layers 28 36
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


CHECKPOINT_EVERY = 25  # rows between in-flight checkpoint writes

from config import COT_INSTRUCTION, LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP
from run_attention_analysis import get_changed_positions
from run_noop_dla import OUT_DIR  # share dla_analysis/ root
from utils.noop_utils import (
    _COT_SWAP_SOURCE_LABELS,
    load_cot_swap_as_patching_pairs,
    load_model,
)


COT_SWAP_DLA_CONTRASTS = (
    "filler_df_correct_vs_noop_clean_wrong",
    "filler_correct_vs_noop_clean_wrong",
    "filler_df_clean_vs_noop_clean_fail",
    "filler_clean_vs_noop_clean_fail",
)

SIDES = ("filler", "noop")
SIDE_COLORS = {"filler": "#2166ac", "noop": "#d6604d"}


def layer_dir(model_name, contrast, layer, cot_source=None):
    """{OUT_DIR}/{model}/{contrast[_cot-{cot_source}]}/l{layer}/.

    For hybrid runs where the donor CoT comes from a different cot_swap
    contrast (e.g. `--cot_source noop_clean`), append the `_cot-{cot_source}`
    suffix to keep results separated from the default (contrast's own CoT).
    Mirrors the patching script's convention.
    """
    contrast_dir = contrast
    if cot_source is not None and cot_source != contrast:
        contrast_dir = f"{contrast}_cot-{cot_source}"
    return OUT_DIR / model_name / contrast_dir / f"l{layer}"


def _checkpoint_path(model_name, contrast, cot_source):
    """Single pickle holding all in-flight accumulator state so a requeued job
    can resume from the last checkpoint instead of restarting from scratch.
    Lives one directory above the per-layer outputs so it's shared across all
    requested layers."""
    contrast_dir = contrast
    if cot_source is not None and cot_source != contrast:
        contrast_dir = f"{contrast}_cot-{cot_source}"
    return OUT_DIR / model_name / contrast_dir / "_inflight_checkpoint.pkl"


def _save_checkpoint(path, accum, row_ids_by_side, n_skip, completed_rows):
    """Atomic checkpoint write via .tmp + rename. Keeps the file readable even
    if the writer is killed mid-pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(
            {
                "accum": accum,
                "row_ids_by_side": row_ids_by_side,
                "n_skip": n_skip,
                "completed_rows": completed_rows,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.rename(path)


def _load_checkpoint(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"warning: could not load checkpoint at {path}: {e!r}")
        return None


def _format_prompt(question, cot_prefix, tokenizer):
    msgs = [{"role": "user", "content": f"{question}\n{COT_INSTRUCTION}"}]
    base = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return base + cot_prefix


def _w_o_per_head(model, block_idx, n_heads, head_dim, d_model):
    W_O_cpu = model.model.layers[block_idx].self_attn.o_proj.weight.detach().float().cpu()
    return W_O_cpu.view(d_model, n_heads, head_dim).permute(1, 0, 2).contiguous()


def run_cot_swap_dla(model, tokenizer, model_name, contrast, layers,
                     cot_source=None):
    """`layers` are paper-display layer indices (1-indexed).

    `cot_source` (default None → use the contrast's own CoT): when set to a
    different cot_swap contrast, the source_cot_prefix is replaced with that
    contrast's symbolic reasoning, mirroring the patching hybrid setup
    (e.g. `--cot_source noop_clean` for filler-DF cliff DLA).
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = model.config
    n_heads, n_kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    # Prefer explicit cfg.head_dim — gemma-2-9b has head_dim=256 while
    # hidden_size//n_heads = 3584//16 = 224, so the derived value is wrong.
    head_dim  = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    kv_groups = n_heads // n_kv_heads
    d_model   = cfg.hidden_size

    block_idxs = [l - 1 for l in layers]
    W_O = {bi: _w_o_per_head(model, bi, n_heads, head_dim, d_model) for bi in block_idxs}

    captured = {}
    hooks = []
    for bi in block_idxs:
        attn_mod = model.model.layers[bi].self_attn

        def make_hook(idx):
            def _h(_module, _inp, out):
                captured[idx] = out.detach()
            return _h

        hooks.append(attn_mod.v_proj.register_forward_hook(make_hook(bi)))

    df = load_cot_swap_as_patching_pairs(source=contrast, cot_source=cot_source)
    cot_note = (
        f" (CoT from `{cot_source}`)"
        if cot_source is not None and cot_source != contrast
        else ""
    )
    print(f"Loaded {len(df)} pairs for {contrast}{cot_note}")

    tok_hash_ids = tokenizer.encode("####", add_special_tokens=False)
    if len(tok_hash_ids) != 1:
        raise ValueError(f"Expected single-token ####, got {tok_hash_ids}")
    tok_hash = tok_hash_ids[0]

    # accumulators per (layer, side):
    #   row-aggregated: attn_mean / ov_mean / dla_mean (used by the headline plots)
    #   position-resolved cache: attn_pos / v_pos plus row_lengths + row_ids,
    #     so subsequent target-token retargeting can run CPU-only without
    #     re-running a GPU forward.
    #
    # In-flight checkpoint: if a previous attempt died (reaper kill / preemption)
    # mid-loop, restore the partial accumulators and skip rows we already did.
    # The trap in scripts/cot_swap_dla.sh requeues the job; pairing it with
    # this resume logic means each restart only loses work since the last
    # CHECKPOINT_EVERY-row checkpoint, not the whole run.
    chk_path = _checkpoint_path(model_name, contrast, cot_source)
    chk = _load_checkpoint(chk_path)
    if chk is not None:
        accum = chk["accum"]
        row_ids_by_side = chk["row_ids_by_side"]
        n_skip = chk["n_skip"]
        completed_rows = chk["completed_rows"]
        # Defensive: ensure the saved accum has entries for every requested
        # (layer, side). If the user widens --layers on a resumed run, fall
        # back to fresh accumulators rather than silently dropping new layers.
        wanted = {(l, s) for l in layers for s in SIDES}
        if not wanted.issubset(accum.keys()):
            missing = wanted - set(accum.keys())
            print(f"warning: checkpoint missing (layer, side) keys {missing}; "
                  "ignoring checkpoint and starting fresh.")
            chk = None
        else:
            print(f"Resuming from checkpoint at {chk_path}: "
                  f"{len(completed_rows)} rows already done")
    if chk is None:
        accum = {
            (l, s): {
                "attn": [], "ov": [], "dla": [],
                "attn_pos": [], "v_pos": [],
            }
            for l in layers for s in SIDES
        }
        row_ids_by_side = {s: [] for s in SIDES}
        n_skip = {s: 0 for s in SIDES}
        completed_rows = set()

    try:
        model.eval()
        for row_idx, (_, row) in enumerate(
            tqdm(df.iterrows(), total=len(df), desc=contrast)
        ):
            key = (int(row["original_id"]), int(row["instance"]))
            if key in completed_rows:
                continue
            filler_prompt = _format_prompt(row["source_question"], row["source_cot_prefix"], tokenizer)
            noop_prompt   = _format_prompt(row["target_question"], row["source_cot_prefix"], tokenizer)
            filler_ids = tokenizer(filler_prompt, add_special_tokens=False)["input_ids"]
            noop_ids   = tokenizer(noop_prompt,   add_special_tokens=False)["input_ids"]
            filler_changed, noop_changed = get_changed_positions(filler_ids, noop_ids)

            for side in SIDES:
                forward_ids   = filler_ids if side == "filler" else noop_ids
                dist_positions = filler_changed if side == "filler" else noop_changed
                if not dist_positions:
                    n_skip[side] += 1
                    continue
                last_pos = len(forward_ids) - 1

                input_t = torch.tensor([forward_ids]).to(model.device)
                with torch.no_grad():
                    out = model(input_t, output_attentions=True)

                row_ids_by_side[side].append((int(row["original_id"]), int(row["instance"])))

                for layer, bi in zip(layers, block_idxs):
                    attn_w = out.attentions[bi][0].float()
                    v_raw = captured[bi][0]
                    v_at_dist = v_raw[dist_positions].float().cpu()
                    v_kv = v_at_dist.view(len(dist_positions), n_kv_heads, head_dim)
                    v_all = v_kv.repeat_interleave(kv_groups, dim=1)

                    P = len(dist_positions)
                    M_all = torch.einsum("hkd,phd->phk", W_O[bi], v_all)
                    M_batch = M_all.reshape(P * n_heads, d_model)
                    with torch.no_grad():
                        logits_batch = model.lm_head(
                            M_batch.to(device=attn_w.device,
                                       dtype=model.model.layers[bi].self_attn.o_proj.weight.dtype)
                        )

                    attn_to_dist_pos = attn_w[:, last_pos, :][:, dist_positions].transpose(0, 1).cpu()
                    attn_to_dist = attn_to_dist_pos.sum(dim=0)

                    ov_per_pos = logits_batch[:, tok_hash].cpu().view(P, n_heads)
                    dla_all = (attn_to_dist_pos * ov_per_pos).sum(dim=0)
                    ov_all = torch.where(
                        attn_to_dist.abs() > 1e-12,
                        dla_all / attn_to_dist,
                        ov_per_pos.mean(dim=0),
                    )
                    accum[(layer, side)]["attn"].append(attn_to_dist.numpy())
                    accum[(layer, side)]["ov"].append(ov_all.numpy())
                    accum[(layer, side)]["dla"].append(dla_all.numpy())

                    # Target-independent slices for CPU retargeting.
                    # Store V at clause positions in the KV-grouped form so
                    # the retargeting script can expand to n_heads itself.
                    accum[(layer, side)]["attn_pos"].append(
                        attn_to_dist_pos.numpy().astype(np.float32)  # (P, n_heads)
                    )
                    accum[(layer, side)]["v_pos"].append(
                        v_kv.numpy().astype(np.float32)              # (P, n_kv_heads, head_dim)
                    )

                del out
                torch.cuda.empty_cache()
            # Mark this row as done AFTER both sides have been processed (or
            # skipped due to empty clause spans). Periodically persist the
            # accumulator state so a reaper kill only costs us the last
            # CHECKPOINT_EVERY rows instead of the whole run.
            completed_rows.add(key)
            if (row_idx + 1) % CHECKPOINT_EVERY == 0:
                _save_checkpoint(
                    chk_path, accum, row_ids_by_side, n_skip, completed_rows
                )
    finally:
        for h in hooks:
            h.remove()
        # Final flush so the most-recent (sub-CHECKPOINT_EVERY) rows survive
        # an exception or kill that doesn't fire the trap (e.g. OOM). On
        # clean completion we delete this file below.
        _save_checkpoint(
            chk_path, accum, row_ids_by_side, n_skip, completed_rows
        )

    for side in SIDES:
        if n_skip[side]:
            print(f"  {side}: skipped {n_skip[side]} rows with empty clause span")

    # Persist per (layer, side). The npz bundles the row-aggregated DLA arrays
    # (for the headline plots, target-locked at ####) with the target-independent
    # per-row position-resolved slices (so retargeting can run CPU-only).
    for layer in layers:
        out_dir = layer_dir(model_name, contrast, layer, cot_source=cot_source)
        out_dir.mkdir(parents=True, exist_ok=True)
        for side in SIDES:
            buf = accum[(layer, side)]
            if not buf["attn"]:
                print(f"  L{layer}/{side}: no rows, skipping save")
                continue
            attn_arr = np.array(buf["attn"])
            ov_arr   = np.array(buf["ov"])
            dla_arr  = np.array(buf["dla"])
            result = {
                "attn_mean": attn_arr.mean(0).tolist(),
                "ov_mean":   ov_arr.mean(0).tolist(),
                "dla_mean":  dla_arr.mean(0).tolist(),
                "target":    "####",
                "target_id": int(tok_hash),
                "side":      side,
                "n":         int(attn_arr.shape[0]),
                "contrast":  contrast,
                "layer":     int(layer),
            }
            (out_dir / f"{side}.json").write_text(json.dumps(result, indent=2))

            row_lengths = np.array([a.shape[0] for a in buf["attn_pos"]], dtype=np.int32)
            attn_pos_concat = np.concatenate(buf["attn_pos"], axis=0)  # (sum_P, n_heads)
            v_pos_concat    = np.concatenate(buf["v_pos"],    axis=0)  # (sum_P, n_kv_heads, head_dim)
            row_ids_arr = np.array(row_ids_by_side[side], dtype=np.int64)  # (N, 2): (original_id, instance)

            np.savez_compressed(
                out_dir / f"{side}_raw.npz",
                attn=attn_arr, ov=ov_arr, dla=dla_arr,
                # Position-resolved cache for CPU retargeting:
                attn_pos=attn_pos_concat, v_pos=v_pos_concat,
                row_lengths=row_lengths, row_ids=row_ids_arr,
                # Constants to disambiguate without loading the model:
                kv_groups=np.int32(kv_groups), n_heads=np.int32(n_heads),
                n_kv_heads=np.int32(n_kv_heads), head_dim=np.int32(head_dim),
            )
            print(f"  L{layer}/{side}: N={result['n']} → {out_dir / f'{side}.json'}")

    # Clean completion: drop the in-flight checkpoint so the next run starts
    # fresh instead of trying to resume from a finished state.
    if chk_path.exists():
        chk_path.unlink()
        print(f"Removed checkpoint {chk_path}")


def plot_cot_swap_dla(model_name, contrast, layer, cot_source=None):
    out = layer_dir(model_name, contrast, layer, cot_source=cot_source)
    data = {}
    for side in SIDES:
        p = out / f"{side}.json"
        if p.exists():
            data[side] = json.load(open(p))
    if len(data) != 2:
        print(f"  L{layer}: missing sides {set(SIDES) - set(data)}, skipping plot")
        return

    fc, nc = data["filler"], data["noop"]
    filler_label, noop_label = _COT_SWAP_SOURCE_LABELS[contrast]
    heads = np.arange(len(fc["dla_mean"]))
    attn_f, attn_n = np.array(fc["attn_mean"]), np.array(nc["attn_mean"])
    dla_f,  dla_n  = np.array(fc["dla_mean"]),  np.array(nc["dla_mean"])
    attn_diff = attn_f - attn_n
    dla_diff  = dla_f  - dla_n

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    ax = axes[0]
    ax.bar(heads - 0.2, attn_f, width=0.4, color=SIDE_COLORS["filler"], alpha=0.85,
           label=f"{filler_label} (N={fc['n']})")
    ax.bar(heads + 0.2, attn_n, width=0.4, color=SIDE_COLORS["noop"], alpha=0.85,
           label=f"{noop_label} (N={nc['n']})")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title("Attention from readout → clause tokens", fontsize=11)
    ax.set_xlabel("Head")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    ax.bar(heads - 0.2, dla_f, width=0.4, color=SIDE_COLORS["filler"], alpha=0.85,
           label=filler_label)
    ax.bar(heads + 0.2, dla_n, width=0.4, color=SIDE_COLORS["noop"], alpha=0.85,
           label=noop_label)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_title(r"DLA toward ####  via clause attention", fontsize=11)
    ax.set_xlabel("Head")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[2]
    colors = ["#2166ac" if d > 0 else "#d6604d" for d in dla_diff]
    ax.bar(heads, dla_diff, color=colors, alpha=0.9)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    for h in np.argsort(np.abs(dla_diff))[-8:]:
        ax.annotate(str(h), (h, dla_diff[h]),
                    textcoords="offset points",
                    xytext=(0, 4 if dla_diff[h] >= 0 else -10),
                    ha="center", fontsize=8, fontweight="bold")
    ax.set_title(
        rf"$\Delta$DLA(####)  =  {filler_label} − {noop_label}  "
        rf"(blue: {filler_label} writes more toward ####)",
        fontsize=11,
    )
    ax.set_xlabel("Head")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.suptitle(f"cot_swap DLA  |  {contrast}  |  L{layer}  |  {model_name}",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {out / 'comparison.png'}")

    pos_top = [int(h) for h in np.argsort(dla_diff)[::-1] if dla_diff[h] > 0][:10]
    summary = {
        "comparison": f"{filler_label} #### DLA minus {noop_label} #### DLA",
        "layer":      int(layer),
        "contrast":   contrast,
        "model_name": model_name,
        "top_dla_diff_heads": [
            {
                "head":              h,
                "dla_diff":          float(dla_diff[h]),
                f"{filler_label}_dla": float(dla_f[h]),
                f"{noop_label}_dla":   float(dla_n[h]),
                "attn_diff":         float(attn_diff[h]),
            }
            for h in pos_top
        ],
    }
    (out / "instrumental_heads.json").write_text(json.dumps(summary, indent=2))
    print(f"Summary → {out / 'instrumental_heads.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct",
                        choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--contrast", default="filler_df_correct_vs_noop_clean_wrong",
                        choices=list(COT_SWAP_DLA_CONTRASTS))
    parser.add_argument(
        "--layers", type=int, nargs="+",
        default=list(range(22, 37)),
        help="Paper-display layer indices L{layer} (1-indexed). Default: 22..36 "
             "(the patching cliff window).",
    )
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip inference; just regenerate plots from existing JSONs.")
    from run_cot_swap_activation_patching import COT_SWAP_PATCHING_CONTRASTS
    parser.add_argument(
        "--cot_source", default=None,
        choices=sorted(COT_SWAP_PATCHING_CONTRASTS) + [None],
        help="Hybrid run: keep `--contrast`'s questions but swap in donor CoT "
             "from another contrast (e.g. `noop_clean`). Output dir gets a "
             "`_cot-<cot_source>` suffix to keep results separated.",
    )
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]

    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        model.set_attn_implementation("eager")
        run_cot_swap_dla(model, tokenizer, model_name, args.contrast, args.layers,
                         cot_source=args.cot_source)

    for layer in args.layers:
        plot_cot_swap_dla(model_name, args.contrast, layer,
                          cot_source=args.cot_source)
