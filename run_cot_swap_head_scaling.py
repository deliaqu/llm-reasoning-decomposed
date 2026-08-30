"""
Self-scaling intervention on cot_swap head sets.

For each (layer, head) cell in a chosen set, multiply the recipient's
per-head output at the readout position by scalar α:

    z^{(L)}[r, h, :]  <-  α * z^{(L)}[r, h, :]

α=0 zero-ablates the head's readout-position contribution; α>1 amplifies it;
α=1 is a no-op control. No donor needed (recipient = noop side).

Outputs the same normalized recovery metric as run_cot_swap_head_patching.py,
so curves can be overlaid on the head-patching bar plot.
"""

import argparse
import json
import hashlib
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP
from run_cot_swap_activation_patching import (
    COT_SWAP_PATCHING_CONTRASTS, build_prompt_and_question_span, encode_prompt,
)
from run_cot_swap_head_patching import (
    DLA_MODES, cells_from_dla, parse_cells, _cells_signature,
    _generate_and_score, compute_pair_baselines,
)
from utils.noop_utils import (
    filter_flipped_pairs,
    load_cot_swap_as_patching_pairs,
    load_model,
    set_cot_swap_model_suffix,
)

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "cot_swap_head_scaling"


def scale_one(model, tokenizer, row, head_cells, n_heads, head_dim, scale,
              side="noop", measure_final_answer=True, max_new_tokens=1024,
              baselines=None, patch_position="cot_end",
              all_positions=False, generation_only=False):
    """Multiply the per-head readout output at every (L, h) cell by scale.

    `patch_position` selects how the prompt is built (cot_end, cot_start,
    mid_cot_<X>); the scaling kernel itself touches only the recipient.

    Hook position-scope (mutually exclusive, default = single-position prefill):
      all_positions=False, generation_only=False (default)
        Scale only at the LAST prompt token during prefill, skip every
        generation step. Useful for log P(####) readouts at cot_end —
        matches the original Fig 6 setup.

      all_positions=True
        Scale at EVERY prompt token during prefill AND at every generation
        step. The head's contribution is α-scaled throughout the entire
        forward + autoregressive rollout (prompt is "processed differently"
        AND generation writes are α-scaled). Use to test whether final-
        answer behaviour changes when the head set is globally disabled.

      generation_only=True
        Scale at the readout slot of prefill (so the first-generated-token
        logits — which come from the prefill last-position residual under
        HF generation — reflect the scaled head) AND at every subsequent
        generation step (seq=1). Positions 0..last_pos-1 of the prompt are
        left unscaled: the model's prompt processing is unmodified except
        at the very slot it uses to commit to the next token. Isolates the
        head's contribution to GENERATED tokens (including the first one)
        from its contribution to processing the bulk of the prompt.

    Returns log P(####) on `side` (noop by default), plus the source/target
    baselines so we can normalize against the filler ceiling.

    Pass a precomputed `baselines` dict (from `compute_pair_baselines` with
    the matching `patch_position`) to skip the redundant source/target
    forward passes across the scale sweep.
    """
    if baselines is None:
        baselines = compute_pair_baselines(model, tokenizer, row, patch_position)
    if baselines.get("patch_position", "cot_end") != patch_position:
        raise ValueError(
            f"baselines patch_position={baselines.get('patch_position')!r} "
            f"!= requested {patch_position!r}; rebuild baselines."
        )
    src_in   = baselines["src_in"]
    tgt_in   = baselines["tgt_in"]
    src_lp   = baselines["src_lp"]
    tgt_lp   = baselines["tgt_lp"]
    denom    = baselines["denom"]
    tok_hash = baselines["tok_hash"]

    def hash_lp(inputs):
        with torch.no_grad():
            out = model(**inputs)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        return lp[0, tok_hash].item()

    # Scale recipient's per-head output at the last token.
    recv_in = tgt_in if side == "noop" else src_in
    recv_last = baselines["tgt_last"] if side == "noop" else baselines["src_last"]

    cells_by_bi = {}
    for L, h in head_cells:
        cells_by_bi.setdefault(L - 1, []).append(h)

    if all_positions and generation_only:
        raise ValueError(
            "all_positions and generation_only are mutually exclusive — "
            "pass at most one."
        )

    def make_scale_hook(heads, last_pos, alpha):
        def hook(_m, inputs):
            x = inputs[0].clone()
            B, seq, hidden = x.shape
            x_view = x.view(B, seq, n_heads, head_dim)
            # Distinguish prefill (seq = full prompt length) from autoregressive
            # generation step (seq = 1, the newly-decoded token). All our
            # prompts are >> 1 token so this is a safe discriminator.
            #
            # ASSUMES KV cache is on (use_cache=True in model.generate). Without
            # the cache, each generation step would re-feed a growing prefix
            # (seq > 1), and the mode-routing below would mis-fire — for
            # `generation_only` the hook would never fire; for `all_positions`
            # it would re-scale the entire prefix on every step. We set
            # use_cache=True explicitly in _generate_and_score (imported from
            # run_cot_swap_head_patching) so this assumption holds.
            is_gen_step = (seq == 1)
            if generation_only:
                # Scale at the readout-position of prefill (so the FIRST
                # generated token, whose logits come from the prefill's
                # last-position residual, reflects the scaled head) AND
                # at every subsequent generation step (seq=1). The other
                # prefill positions (0..last_pos-1) are left unscaled, so
                # the model's prompt-processing is unmodified except at
                # the slot it uses to predict the next token.
                if is_gen_step:
                    for h in heads:
                        x_view[:, :, h, :] = x_view[:, :, h, :] * alpha
                else:
                    if seq <= last_pos:
                        return None
                    for h in heads:
                        x_view[:, last_pos, h, :] = x_view[:, last_pos, h, :] * alpha
            elif all_positions:
                # Scale at every position (prefill + gen).
                for h in heads:
                    x_view[:, :, h, :] = x_view[:, :, h, :] * alpha
            else:
                # Single-position mode: scale only at the readout token of
                # prefill, skip every generation step.
                if seq <= last_pos:
                    return None
                for h in heads:
                    x_view[:, last_pos, h, :] = x_view[:, last_pos, h, :] * alpha
            return (x_view.view(B, seq, hidden),)
        return hook

    handles = []
    extra = {}
    try:
        for bi, heads in cells_by_bi.items():
            m = model.model.layers[bi].self_attn.o_proj
            handles.append(m.register_forward_pre_hook(
                make_scale_hook(heads, recv_last, scale)
            ))
        if measure_final_answer:
            gold = row.get("target_answer") if side == "noop" \
                else row.get("source_answer")
            gen = _generate_and_score(
                model, tokenizer, recv_in, gold_answer=gold,
                tok_hash=tok_hash, max_new_tokens=max_new_tokens,
            )
            scaled_lp = gen["patched_hash_logprob"]
            extra = {
                "gold_answer":                  str(gold) if gold is not None else None,
                "scaled_hash_emitted":          gen["patched_hash_emitted"],
                "scaled_final_answer_success":  gen["patched_final_answer_success"],
                "scaled_gold_logprob":          gen["patched_gold_logprob"],
                "scaled_gold_immediate":        gen["patched_gold_immediate"],
                "scaled_emitted_answer":        gen["patched_emitted_answer"],
                "scaled_generation":            gen["patched_generation"],
            }
        else:
            scaled_lp = hash_lp(recv_in) if patch_position == "cot_end" else None
    finally:
        for h in handles:
            h.remove()

    baseline_lp = tgt_lp if side == "noop" else src_lp
    # log P(####) readout is only meaningful at cot_end (model is about to
    # emit ####). At cot_start / mid_cot, src_lp/tgt_lp are None and
    # readout_delta / normalized_effect are undefined.
    if baseline_lp is not None and scaled_lp is not None:
        readout_delta = scaled_lp - baseline_lp
        normalized_effect = (
            readout_delta / denom
            if denom is not None and abs(denom) > 1e-6 else None
        )
    else:
        readout_delta = None
        normalized_effect = None
    return {
        "source_hash_logprob":       src_lp,
        "target_hash_logprob":       tgt_lp,
        "denom_source_minus_target": denom,
        "side":                      side,
        "scale":                     float(scale),
        "patch_position":            patch_position,
        "baseline_hash_logprob":     baseline_lp,
        "scaled_hash_logprob":       scaled_lp,
        "readout_delta":             readout_delta,
        "normalized_effect":         normalized_effect,
        "num_cells":                 int(len(head_cells)),
        **extra,
    }


def _salvage_from_filler_df_sibling(out_dir, out_path, cells_sig, label,
                                    pair_source, patch_position):
    """Pre-populate `out_path` with rows from a filler_df_correct sibling
    run (same cells, same scope) when the current run is on a self-paired
    source whose recipient prompt matches the sibling's recipient.

    Why this is safe: at cot_start --all_positions/--genonly/sinpos, the
    scaling hook fires only on the *recipient* prompt. The source-side prompt
    is built but unused (`denom_source_minus_target=-inf`, gap filter bypassed).
    For matching (oid, instance, scale) × matching cells signature, recipient
    processing is bit-identical across pair sources, so the salvaged rows
    are interchangeable with re-computed ones.

    Salvage map:
      - noop_clean_wrong_self  ↔  filler_df_correct_vs_noop_clean_wrong side=noop
        (recipient = noop_question; sibling jsonl = noop.jsonl)
      - filler_correct_self    ↔  filler_df_correct_vs_noop_clean_wrong side=filler
        (recipient = filler-DF question; sibling jsonl = filler.jsonl)

    Only acts at patch_position == "cot_start" with a matching sibling dir
    whose head_cells.json signature equals the current run's.

    Returns the number of rows salvaged (0 if no salvage performed).
    """
    if patch_position != "cot_start":
        return 0

    if pair_source == "noop_clean_wrong_self":
        marker = "_pair-noop_clean_wrong_self"
        sibling_jsonl_name = "noop.jsonl"
        side_label_value = "noop_clean_wrong"
    elif pair_source == "filler_correct_self":
        marker = "_pair-filler_correct_self"
        sibling_jsonl_name = "filler.jsonl"
        side_label_value = "filler_correct"
    else:
        return 0

    if marker not in label:
        return 0
    # Mirror the canonical filler_df_correct sibling naming:
    #   <base>_pair-noop_clean_wrong_self   ↔   <base>_cot-noop_clean
    #   <base>_pair-filler_correct_self     ↔   <base>_cot-noop_clean
    # The pos_tag suffix is the same (e.g. _cot_start_allpos_nofl).
    pos_tag_part = out_dir.name[len(label):]  # everything after label is pos_tag
    sibling_label = label.replace(marker, "_cot-noop_clean")
    sibling_dir = out_dir.parent / f"{sibling_label}{pos_tag_part}"
    sibling_jsonl = sibling_dir / sibling_jsonl_name
    sibling_cells = sibling_dir / "head_cells.json"
    if not (sibling_jsonl.exists() and sibling_cells.exists()):
        return 0
    sib_sig = json.load(open(sibling_cells)).get("cells_signature")
    if sib_sig != cells_sig:
        return 0  # different cells - cannot safely salvage

    # Collect keys already in the current output to avoid duplicates.
    done = set()
    if out_path.exists():
        for line in open(out_path):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("cells_signature") != cells_sig:
                continue
            done.add((int(r["original_id"]), int(r["instance"]),
                      float(r["scale"])))

    appended = 0
    with open(out_path, "a") as f_out:
        for line in open(sibling_jsonl):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("cells_signature") != cells_sig:
                continue
            key = (int(r["original_id"]), int(r["instance"]), float(r["scale"]))
            if key in done:
                continue
            # Rewrite metadata fields that depend on pair_source so the
            # salvaged row looks like it was produced under this contrast.
            # All computed fields (scaled_*, denom, baselines) are unchanged
            # because they depend only on the recipient prompt.
            r["pair_source"] = pair_source
            r["cot_source"] = pair_source
            r["source_label"] = side_label_value
            old_label = r.get("label", "")
            if "_cot-noop_clean" in old_label:
                r["label"] = old_label.replace("_cot-noop_clean", marker)
            f_out.write(json.dumps(r) + "\n")
            done.add(key)
            appended += 1
    if appended:
        print(f"Salvaged {appended} rows from sibling "
              f"{sibling_dir.name} (same cells_signature={cells_sig})")
    return appended


def run(model, tokenizer, model_name, contrast, head_cells, label,
        scale_factors, side="noop", min_source_target_gap=0.5, overwrite=False,
        measure_final_answer=True, max_new_tokens=1024,
        pair_source=None, cot_source=None, patch_position="cot_end",
        all_positions=False, filter_flipped=False, generation_only=False,
        max_pairs_per_template=None):
    """`contrast` controls DLA cell lookup + output directory. `pair_source`
    is the cot_swap CSV for the (source_q, target_q) pair (defaults to
    `contrast`). `cot_source` is where the donor CoT comes from (defaults to
    `pair_source`); set it to a different contrast to combine `pair_source`'s
    questions with another contrast's CoT.

    `patch_position` (default cot_end) selects where the scaling hook fires.
    cot_start variants get a `_cot_start` suffix on the output directory.

    `all_positions` (default False): when True, the scaling hook fires at
    every prompt position during prefill AND every newly-generated token
    (global α-scaling of the head set). Output dir gets an `_allpos` suffix
    so single-position and global-scale runs don't collide.
    """
    if pair_source is None:
        pair_source = contrast
    if cot_source is None:
        cot_source = pair_source
    out_dir = OUT_DIR / model_name / contrast / f"{label}{_pos_tag(patch_position, all_positions, filter_flipped, generation_only)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{side}.jsonl"
    cells_path = out_dir / "head_cells.json"
    cells_sig = _cells_signature(head_cells)

    if cells_path.exists():
        prev = json.load(open(cells_path))
        prev_sig = prev.get("cells_signature")
        if prev_sig != cells_sig and out_path.exists() and not overwrite:
            raise SystemExit(
                f"head_cells mismatch under label '{label}': "
                f"existing sig={prev_sig}, new sig={cells_sig}. "
                f"Pass --overwrite or use a different --label."
            )

    if overwrite and out_path.exists():
        out_path.unlink()
    cells_path.write_text(json.dumps({
        "head_cells": head_cells, "num_cells": len(head_cells),
        "cells_signature": cells_sig,
    }, indent=2))

    # Auto-salvage: pre-populate out_path with rows from the filler_df_correct
    # sibling run when this run is noop_clean_wrong_self @ cot_start (same
    # cells, same scope flags). At cot_start the source-side prompt is unused,
    # so any (oid, instance, scale) row computed for the filler_df_correct
    # pair source is bit-identical to what we'd compute here for the same
    # key. Saves recomputation on the ~497 pairs that overlap between the
    # two pair sources.
    if not overwrite:
        _salvage_from_filler_df_sibling(
            out_dir, out_path, cells_sig, label, pair_source, patch_position
        )

    # Resume — when `measure_final_answer` is on, rows without the new field
    # are dropped from the file and re-run (avoids duplicates after the
    # upgrade pass).
    done = set()
    if out_path.exists():
        n_upgrade = 0
        keep_lines = []
        for line in open(out_path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("cells_signature") != cells_sig:
                continue
            if measure_final_answer and "scaled_final_answer_success" not in r:
                n_upgrade += 1
                continue
            keep_lines.append(line if line.endswith("\n") else line + "\n")
            done.add((int(r["original_id"]), int(r["instance"]),
                      float(r["scale"])))
        if n_upgrade:
            out_path.write_text("".join(keep_lines))
            print(f"Resuming — {len(done)} (row, scale) cells already done; "
                  f"{n_upgrade} rows lacked final-answer fields and were dropped.")
        else:
            print(f"Resuming — {len(done)} (row, scale) cells already done")

    cfg = model.config
    n_heads = cfg.num_attention_heads
    # Prefer explicit cfg.head_dim — gemma-2-9b has head_dim=256 while
    # hidden_size//n_heads = 3584//16 = 224, so the derived value is wrong.
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)

    df = load_cot_swap_as_patching_pairs(source=pair_source, cot_source=cot_source)
    if filter_flipped:
        before = len(df)
        df = filter_flipped_pairs(df)
        print(f"filter_flipped: dropped {before - len(df)} flipped pairs → {len(df)}")
    if max_pairs_per_template is not None and max_pairs_per_template > 0:
        before = len(df)
        df = (
            df.sample(frac=1, random_state=0)
              .groupby("original_id", group_keys=False, sort=False)
              .head(max_pairs_per_template)
              .sort_values(["original_id", "instance"])
              .reset_index(drop=True)
        )
        print(f"max_pairs_per_template={max_pairs_per_template}: "
              f"{before} → {len(df)} pairs ({df['original_id'].nunique()} templates)")
    # anchor_clause: precompute per-row divergence-DLA anchor token index and
    # drop rows where no anchor can be found. Done once here (not per-scale)
    # so the run loop can use row["_anchor_idx"] downstream without re-running
    # the (relatively cheap) anchor logic on every iteration.
    if patch_position == "anchor_clause":
        from run_cot_swap_dla_divergence import find_anchor as _find_anchor
        anchors = []
        n_skipped = 0
        for _, r in df.iterrows():
            info = _find_anchor(r, tokenizer, num_strategy="set_diff")
            anchors.append(info[0] if info is not None else None)
            if info is None:
                n_skipped += 1
        df["_anchor_idx"] = anchors
        df = df[df["_anchor_idx"].notna()].reset_index(drop=True)
        print(f"anchor_clause: {len(df)} pairs anchored ({n_skipped} skipped)")
    cot_note = f"; CoT from `{cot_source}`" if cot_source != pair_source else ""
    print(f"Loaded {len(df)} pairs from `{pair_source}`{cot_note}; "
          f"{len(head_cells)} cells (DLA from `{contrast}`); "
          f"scales={scale_factors}; side={side}")
    model.eval()

    with open(out_path, "a") as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
            # All scales for a row share the same (src, tgt) baselines, so
            # compute them once. Skip the whole row if the gap is below the
            # filter threshold — avoids N_scales × (2 baseline forwards + 1
            # generate) of wasted work on filtered rows.
            row_scales = [
                s for s in scale_factors
                if (int(row["original_id"]), int(row["instance"]), float(s))
                not in done
            ]
            if not row_scales:
                continue
            baselines = compute_pair_baselines(
                model, tokenizer, row, patch_position=patch_position
            )
            if baselines.get("mid_cot_too_short"):
                continue
            # Gap-filter only applies at cot_end (where log P(####) is the
            # actual readout). cot_start / mid_cot_X skip the baseline
            # forward passes; denom is sentinel -inf there, so don't compare.
            if patch_position == "cot_end" and baselines["denom"] < min_source_target_gap:
                continue
            for s in row_scales:
                metrics = scale_one(
                    model, tokenizer, row, head_cells,
                    n_heads=n_heads, head_dim=head_dim,
                    scale=float(s), side=side,
                    measure_final_answer=measure_final_answer,
                    max_new_tokens=max_new_tokens,
                    baselines=baselines,
                    patch_position=patch_position,
                    all_positions=all_positions,
                    generation_only=generation_only,
                )
                rec = {
                    "original_id":     int(row["original_id"]),
                    "instance":        int(row["instance"]),
                    "contrast":        contrast,
                    "pair_source":     pair_source,
                    "cot_source":      cot_source,
                    "label":           label,
                    "cells_signature": cells_sig,
                    "source_label":    row["source_label"],
                    "target_label":    row["target_label"],
                    **metrics,
                }
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()

    print(f"Results → {out_path}")


def _pos_tag(patch_position, all_positions=False, filter_flipped=False,
             generation_only=False):
    if patch_position == "cot_end":
        pos_tag = "_cot_end_eol"
    else:
        pos_tag = f"_{patch_position}"
    if all_positions:
        pos_tag += "_allpos"
    if generation_only:
        pos_tag += "_genonly"
    if filter_flipped:
        pos_tag += "_nofl"
    return pos_tag


def plot(model_name, contrast, label, side="noop", min_gap=0.5, n_boot=2000,
         patch_position="cot_end", all_positions=False, filter_flipped=False,
         generation_only=False):
    out_dir = OUT_DIR / model_name / contrast / f"{label}{_pos_tag(patch_position, all_positions, filter_flipped, generation_only)}"
    # The built-in plotter shows normalized log P(####) recovery vs α —
    # only meaningful at cot_end where log P(####) is the actual readout.
    # For cot_start / mid_cot_X / *_allpos / *_genonly, the JSONL rows
    # intentionally have normalized_effect = None (no #### baseline was
    # computed), so this plot is a no-op. Behavioral plots (success rate +
    # gold logP) for those positions come from plot_recovery_sweep.py.
    if patch_position != "cot_end":
        print(f"Skipping built-in log-P(####) plot for patch_position={patch_position!r}.")
        print(f"  Use: python interpretability/plot_recovery_sweep.py")
        print(f"  for behavioral metrics on this output dir: {out_dir}")
        return
    path = out_dir / f"{side}.jsonl"
    if not path.exists():
        print(f"No results at {path}")
        return
    raw = [json.loads(l) for l in open(path) if l.strip()]
    seen = {}
    for r in raw:
        seen[(r["original_id"], r["instance"], r["scale"])] = r
    rows = [r for r in seen.values()
            if r.get("normalized_effect") is not None
            and float(r.get("denom_source_minus_target", 0.0)) >= min_gap]
    if not rows:
        print("No usable rows.")
        return

    by_scale = {}
    for r in rows:
        by_scale.setdefault(float(r["scale"]), []).append(r)

    scales = sorted(by_scale)
    means, los, his, ns = [], [], [], []
    rng = np.random.default_rng(0)
    for s in scales:
        rs = by_scale[s]
        eff = np.array([r["normalized_effect"] for r in rs])
        clust = np.array([r["original_id"] for r in rs])
        uniq, inv = np.unique(clust, return_inverse=True)
        per_t = np.array([eff[inv == i].mean() for i in range(len(uniq))])
        boots = np.array([
            per_t[rng.integers(0, len(per_t), len(per_t))].mean()
            for _ in range(n_boot)
        ])
        means.append(float(per_t.mean()))
        los.append(float(np.quantile(boots, 0.025)))
        his.append(float(np.quantile(boots, 0.975)))
        ns.append(len(rs))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(scales, means, marker="o", color="#2166ac", lw=1.6)
    ax.fill_between(scales, los, his, color="#2166ac", alpha=0.2)
    ax.axhline(0, color="black", lw=0.7, alpha=0.6, label="noop baseline")
    ax.axhline(1, color="black", lw=0.7, ls="--", alpha=0.4,
               label="filler ceiling (Fig. 6)")
    ax.axvline(1.0, color="gray", lw=0.5, ls=":", alpha=0.6)
    ax.set_xlabel(r"scale factor $\alpha$")
    ax.set_ylabel("Normalized recovery $\\rho$")
    ax.set_title(f"{label}  ({ns[0]} rows / scale)\n"
                 f"{contrast}", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / f"{side}_scaling.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out_png}")
    for s, m, lo, hi, n in zip(scales, means, los, his, ns):
        print(f"  α={s:>4.2f}:  ρ={m:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  n_rows={n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct",
                        choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--contrast", default="filler_df_correct_vs_noop_clean_wrong",
                        choices=list(COT_SWAP_PATCHING_CONTRASTS),
                        help="Used for the DLA cell-set lookup and the output "
                             "directory. Decoupled from --pair_source.")
    parser.add_argument("--pair_source", default=None,
                        choices=list(COT_SWAP_PATCHING_CONTRASTS),
                        help="Pair-data source for (source_q, target_q). "
                             "Defaults to --contrast.")
    parser.add_argument("--cot_source", default=None,
                        choices=list(COT_SWAP_PATCHING_CONTRASTS),
                        help="Source of the donor CoT. Defaults to "
                             "--pair_source. Set to `noop_clean` for "
                             "symbolic's clean CoT while keeping pair "
                             "questions length-matched.")
    cells = parser.add_mutually_exclusive_group(required=False)
    cells.add_argument("--cells", type=str)
    cells.add_argument("--cells_from_dla", action="store_true")
    parser.add_argument("--dla_threshold", type=float, default=1e-4)
    parser.add_argument("--dla_mode", default="pro_noop", choices=list(DLA_MODES),
                        help="Default pro_noop: amplify the pro-#### channel. "
                             "Use h_minus_div / h_plus_div / h_combined_sym_diff "
                             "with --dla_contrast for engagement-anchored DLA.")
    parser.add_argument("--dla_contrast", type=str, default=None,
                        help="Override the DLA directory contrast (defaults to "
                             "--contrast). Set to e.g. "
                             "filler_df_correct_vs_noop_clean_wrong_divergence_dla_nofl "
                             "to read engagement-anchored DLA cells while keeping "
                             "--contrast as the pair-data contrast.")
    parser.add_argument("--min_noop_attn", type=float, default=0.0)
    parser.add_argument("--layers", type=int, nargs="+",
                        default=list(range(22, 37)))
    parser.add_argument("--scale_factors", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    parser.add_argument("--side", default="noop", choices=["noop", "filler"])
    parser.add_argument("--min_source_target_gap", type=float, default=0.5)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--measure_final_answer", action="store_true", default=True,
                        help="Always on (kept for backward compatibility). "
                             "whether the model emits the gold final answer. "
                             "Slower but gives behavioral (not just log P(####)) "
                             "recovery.")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Generation budget when --measure_final_answer. "
                             "Early-stops at the first numeric token after "
                             "`####`, so the budget is only the hard ceiling "
                             "for runaways that never emit `####`.")
    parser.add_argument("--patch_position", default="cot_end",
                        help="Prefill slot at which to scale per-head V. "
                             "cot_end = last token before #### (default). "
                             "cot_start = last token of [question + chat_template] "
                             "(no donor CoT); model generates own CoT after the "
                             "scaled prefill. "
                             "mid_cot_<X> = X tokens into each side's CoT "
                             "(donor = symbolic, recipient = noop's own; "
                             "rejects pairs where either CoT is shorter than X).")
    parser.add_argument("--all_positions", action="store_true",
                        help="If set, scale the head set at every prompt "
                             "position during prefill AND every generated "
                             "token (global α-scaling). Default is "
                             "single-position at the readout slot. "
                             "Output dir gets `_allpos` suffix so it does "
                             "not collide with single-position runs.")
    parser.add_argument("--filter_flipped", action="store_true",
                        help="Drop pairs where noop's natural re-generation "
                             "produces the gold answer (despite being labeled "
                             "noop_clean_wrong). Output dir gets `_nofl` "
                             "suffix.")
    parser.add_argument("--max_pairs_per_template", type=int, default=None,
                        help="Cap pairs per original_id (template) to this many, "
                             "drawn via fixed-seed shuffle. Stratified subsample "
                             "for fast ablation/recovery runs without losing "
                             "any templates.")
    parser.add_argument("--generation_only", action="store_true",
                        help="Scale the head set at the prefill readout slot "
                             "that predicts the first generated token, then "
                             "at each subsequent generation step. Earlier "
                             "prompt positions are left unscaled. "
                             "Mutually exclusive with --all_positions. "
                             "Output dir gets `_genonly` suffix.")
    args = parser.parse_args()
    if args.all_positions and args.generation_only:
        raise SystemExit("--all_positions and --generation_only are mutually exclusive.")

    model_name = MODEL_NAME_MAP[args.model_id]
    # Inference/cot_swap CSVs are named by the HF model basename; keep the
    # data loaders pointed at the model actually being run (not the default).
    set_cot_swap_model_suffix(args.model_id)

    if args.cells_from_dla:
        head_cells = cells_from_dla(
            model_name, args.contrast, args.layers,
            threshold=args.dla_threshold, mode=args.dla_mode,
            min_noop_attn=args.min_noop_attn,
            dla_contrast=args.dla_contrast,
        )
    elif args.cells:
        head_cells = parse_cells(args.cells)
    else:
        raise SystemExit("Pass --cells or --cells_from_dla.")

    # Auto-label mirrors the head-patching naming
    if args.label:
        label = args.label
    elif args.cells_from_dla:
        prefix = {"wrong_direction": "wrong_dir", "filler_supporting": "filler_supp",
                  "combined": "combined", "pro_noop": "pro_noop",
                  "h_minus_div": "h_minus_div", "h_plus_div": "h_plus_div",
                  "h_combined_sym_diff": "h_combined_sym_diff"}[args.dla_mode]
        label = f"{prefix}_thr_{args.dla_threshold:.0e}".replace("-0", "-")
        if args.min_noop_attn > 0:
            label += f"_attn{args.min_noop_attn:g}"
    else:
        label = "custom_cells"

    pair_source = args.pair_source or args.contrast
    cot_source = args.cot_source or pair_source
    if pair_source != args.contrast:
        label = f"{label}_pair-{pair_source}"
    if cot_source != pair_source:
        label = f"{label}_cot-{cot_source}"
    print(f"Cell set label: {label}  ({len(head_cells)} cells)")
    print(f"DLA contrast (output + cell lookup): {args.contrast}")
    print(f"Pair source (questions):             {pair_source}")
    print(f"CoT source (donor reasoning):        {cot_source}")

    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        run(model, tokenizer, model_name, args.contrast, head_cells, label,
            scale_factors=args.scale_factors, side=args.side,
            min_source_target_gap=args.min_source_target_gap,
            overwrite=args.overwrite,
            measure_final_answer=args.measure_final_answer,
            pair_source=pair_source,
            cot_source=cot_source,
            max_new_tokens=args.max_new_tokens,
            patch_position=args.patch_position,
            all_positions=args.all_positions,
            filter_flipped=args.filter_flipped,
            generation_only=args.generation_only,
            max_pairs_per_template=args.max_pairs_per_template)
    plot(model_name, args.contrast, label, side=args.side,
         min_gap=args.min_source_target_gap,
         patch_position=args.patch_position,
         all_positions=args.all_positions,
         filter_flipped=args.filter_flipped,
         generation_only=args.generation_only)
