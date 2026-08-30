"""
Multi-cell head-level activation patching for cot_swap pairs.

For each (source, target) pair, replace the recipient's per-head output at the
readout position (last token of the shared CoT prefix) at a fixed set of
(layer, head) cells with the donor's, then measure log P(next = ####).

Default cell set = the "wrong-direction" heads identified by run_cot_swap_dla:
  heads where the recipient side's dla_mean[h] is large and negative
  (head writes anti-#### via clause attention).

This isolates the candidate cliff mechanism: if patching only these heads
recovers a large fraction of the full-residual cliff in Figure 6, the cliff
is owned by a small, DLA-identifiable head set. Per-cell intervention is at
the o_proj input (post-(softmax(QK)·V), pre-projection), so o_proj's sum
across heads then assembles the modified attention contribution.

Usage:
    python run_cot_swap_head_patching.py \
        --cells_from_dla --dla_threshold 1e-4

    python run_cot_swap_head_patching.py \
        --cells "36:4 36:28 36:42 35:42 34:28"

    python run_cot_swap_head_patching.py --plot_only --label wrong_dir_1em4
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import hashlib
import json
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
from run_cot_swap_dla import layer_dir as dla_layer_dir
from run_noop_activation_patching import (
    _extract_emitted_answer, _is_correct_final_answer,
    AnswerBoundaryStoppingCriteria,
)
from transformers import StoppingCriteriaList
from utils.noop_utils import (
    filter_flipped_pairs,
    load_cot_swap_as_patching_pairs,
    load_model,
)

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "cot_swap_head_patching"


def _generate_and_score(model, tokenizer, recv_in, gold_answer, tok_hash,
                        max_new_tokens=1024):
    """With hooks active, run a single greedy generate() and return five
    metrics — paralleling run_cot_swap_activation_patching.generate_and_score
    so the head_patching outputs are directly comparable with the layer/scope
    patching outputs at the same readout position.

    Returns:
      patched_hash_logprob          — log P(####) at step 0 (continuous).
      patched_hash_emitted          — was `####` the FIRST generated token?
                                       (immediate-#### rate; sparse binary.)
      patched_final_answer_success  — LOOSE binary: did the last number after
                                       any `####` in the generation match
                                       gold? (Saturates ~1.0 because the model
                                       almost always finds the answer in
                                       max_new_tokens.)
      patched_gold_logprob          — log P(gold first-token candidate) at
                                       the post-#### slot (continuous).
      patched_gold_immediate        — TIGHT binary: was a gold first-token
                                       candidate the model's actual next
                                       token after `####`? (Discrete
                                       complement of patched_gold_logprob.)

    Stops at the answer boundary via `AnswerBoundaryStoppingCriteria`.
    """
    from run_noop_activation_patching import (
        _gold_answer_logprob_from_generation_scores,
        _gold_first_token_candidates,
        _boundary_end_index,
    )

    prompt_len = recv_in["input_ids"].shape[1]
    prompt_ids = recv_in["input_ids"][0].detach().cpu().tolist()
    stopping_criteria = StoppingCriteriaList([
        AnswerBoundaryStoppingCriteria(tokenizer, prompt_len)
    ])
    with torch.no_grad():
        out = model.generate(
            **recv_in,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            return_dict_in_generate=True,
            output_scores=True,
            # KV cache MUST be on: the head_scaling `generation_only` hook
            # discriminates prefill vs. generation steps by checking seq == 1,
            # which only holds when each new token is fed alone (with cache).
            # Without the cache, each step would re-feed a growing prefix and
            # the hook's mode-routing would mis-fire. True is HF default for
            # Llama but we set it explicitly so behavior is independent of
            # any future default change.
            use_cache=True,
        )
    step0 = out.scores[0]
    lp = F.log_softmax(step0.float(), dim=-1)
    hash_lp = float(lp[0, tok_hash].item())

    gen_ids_tensor = out.sequences[0, prompt_len:]
    gen_ids = gen_ids_tensor.detach().cpu().tolist()
    generation = tokenizer.decode(gen_ids_tensor, skip_special_tokens=True)
    emitted = _extract_emitted_answer(generation)
    success = (
        gold_answer is not None
        and _is_correct_final_answer(generation, str(gold_answer))
    )
    # Discrete complement of patched_hash_logprob.
    hash_emitted = (
        gen_ids_tensor.numel() > 0
        and int(gen_ids_tensor[0].item()) == tok_hash
    )
    # Gold log-prob at the boundary + its argmax indicator. Mirrors the
    # activation-patching script so the metric semantics are identical.
    gold_lp = float("nan")
    gold_immediate = False
    if gold_answer is not None:
        gold_lp = float(_gold_answer_logprob_from_generation_scores(
            model, tokenizer, prompt_ids, gen_ids,
            str(gold_answer), out.scores,
        ))
        boundary_end = _boundary_end_index(tokenizer, gen_ids)
        if boundary_end is not None and boundary_end < len(gen_ids):
            gold_candidates = set(
                _gold_first_token_candidates(tokenizer, str(gold_answer))
            )
            gold_immediate = int(gen_ids[boundary_end]) in gold_candidates
    return {
        "patched_hash_logprob":          hash_lp,
        "patched_hash_emitted":          bool(hash_emitted),
        "patched_final_answer_success":  bool(success),
        "patched_gold_logprob":          gold_lp,
        "patched_gold_immediate":        bool(gold_immediate),
        "patched_emitted_answer":        emitted,
        "patched_generation":            generation,
    }


def parse_cells(s):
    """Parse 'L:h L:h L,h ...' into [(L, h), ...] of paper-layer indices."""
    out = []
    for tok in re.split(r"[,\s]+", s.strip()):
        if not tok:
            continue
        L, h = tok.split(":")
        out.append((int(L), int(h)))
    return sorted(set(out))


DLA_MODES = ("wrong_direction", "filler_supporting", "combined", "pro_noop",
             "h_minus_div", "h_plus_div", "h_combined_sym_diff")


def cells_from_dla(model_name, contrast, layers, threshold, mode,
                   min_noop_attn=0.0, dla_contrast=None):
    """Pick (L, h) cells from the DLA jsons by `mode`:
      wrong_direction:    heads with noop_dla_mean[h] < -threshold
                          (clause-attention head writes anti-#### on noop)
      filler_supporting:  heads with filler_dla_mean[h] > +threshold
                          (head writes pro-#### on filler; quiet on noop)
      pro_noop:           heads with noop_dla_mean[h] > +threshold
                          (head attends to noop clause and writes pro-####
                          on noop — the candidate rescue channel; mirror of
                          wrong_direction)
      combined:           union of wrong_direction + filler_supporting.

      Engagement-anchored / divergence DLA variants (set
      `dla_contrast="<contrast>_divergence_dla_nofl"` to read from the
      run_cot_swap_dla_divergence outputs):
      h_minus_div:        anti-engagement = noop_dla[h] < -threshold
      h_plus_div:         pro-engagement  = noop_dla[h] > +threshold
      h_combined_sym_diff: union of h_minus_div and h_plus_div, EXCLUDING
                          heads where the filler side's |DLA| also exceeds
                          threshold (sym_diff filter — keep only cells that
                          are noop-specific, not background readout heads).

    `min_noop_attn` AND-filters the result: only keep cells with noop_side
    `attn_mean[h] >= min_noop_attn`. Use this to restrict to cells where the
    clause path is a big enough fraction of the head's readout output that
    swapping it actually moves the readout (instead of being dwarfed by
    non-clause writes).

    `dla_contrast`: optional override for the DLA directory lookup. When
    `None`, defaults to `contrast` (####-anchored cot_swap_dla). Set to the
    engagement-anchored path (e.g. `<contrast>_divergence_dla_nofl`) to read
    from run_cot_swap_dla_divergence outputs while keeping `contrast` set
    to the pair-data contrast.
    """
    lookup_contrast = dla_contrast or contrast

    def _read(L, side, field="dla_mean"):
        p = dla_layer_dir(model_name, lookup_contrast, L) / f"{side}.json"
        if not p.exists():
            print(f"  (skip L{L}/{side}: {p} missing)")
            return None
        return np.array(json.load(open(p))[field])

    cells = set()
    for L in layers:
        noop_attn = _read(L, "noop", "attn_mean") if min_noop_attn > 0 else None
        if mode in ("wrong_direction", "combined"):
            noop_dla = _read(L, "noop")
            if noop_dla is not None:
                for h in range(len(noop_dla)):
                    if noop_dla[h] < -threshold:
                        if noop_attn is None or noop_attn[h] >= min_noop_attn:
                            cells.add((L, int(h)))
        if mode in ("filler_supporting", "combined"):
            filler_dla = _read(L, "filler")
            if filler_dla is not None:
                for h in range(len(filler_dla)):
                    if filler_dla[h] > +threshold:
                        if noop_attn is None or noop_attn[h] >= min_noop_attn:
                            cells.add((L, int(h)))
        if mode == "pro_noop":
            noop_dla = _read(L, "noop")
            if noop_dla is not None:
                for h in range(len(noop_dla)):
                    if noop_dla[h] > +threshold:
                        if noop_attn is None or noop_attn[h] >= min_noop_attn:
                            cells.add((L, int(h)))
        if mode in ("h_minus_div", "h_plus_div", "h_combined_sym_diff"):
            noop_dla = _read(L, "noop")
            filler_dla = _read(L, "filler") if mode == "h_combined_sym_diff" else None
            if noop_dla is None:
                continue
            for h in range(len(noop_dla)):
                hits_minus = noop_dla[h] < -threshold
                hits_plus  = noop_dla[h] > +threshold
                if mode == "h_minus_div" and not hits_minus: continue
                if mode == "h_plus_div"  and not hits_plus:  continue
                if mode == "h_combined_sym_diff":
                    if not (hits_minus or hits_plus): continue
                    # sym_diff: exclude heads where filler-side also writes
                    # strongly (these are background readout heads, not
                    # noop-specific).
                    if filler_dla is not None and abs(filler_dla[h]) > threshold:
                        continue
                if noop_attn is None or noop_attn[h] >= min_noop_attn:
                    cells.add((L, int(h)))
    return sorted(cells)


_LABEL_PREFIX = {
    "wrong_direction":     "wrong_dir",
    "filler_supporting":   "filler_supp",
    "combined":            "combined",
    "pro_noop":            "pro_noop",
    "h_minus_div":         "h_minus_div",
    "h_plus_div":          "h_plus_div",
    "h_combined_sym_diff": "h_combined_sym_diff",
}


def _label_for_cells(label, cells_from_dla, dla_threshold, mode,
                     min_noop_attn=0.0):
    if label:
        return label
    if cells_from_dla:
        prefix = _LABEL_PREFIX[mode]
        base = f"{prefix}_thr_{dla_threshold:.0e}".replace("-0", "-")
        if min_noop_attn > 0:
            base += f"_attn{min_noop_attn:g}"
        return base
    return "custom_cells"


def _truncate_cot_strict(cot, X, tokenizer):
    """Tokenize `cot` and return the first X tokens decoded back to string,
    or None if `cot` is missing/empty/shorter than X tokens. Strict: caller
    must skip the row when this returns None — for mid_cot the whole point
    is "what does the model commit to at exactly X tokens into the CoT",
    so feeding a shorter-than-X full CoT would conflate positions.
    """
    if not isinstance(cot, str) or not cot:
        return None
    ids = tokenizer(cot, add_special_tokens=False)["input_ids"]
    if len(ids) < X:
        return None
    return tokenizer.decode(ids[:X], skip_special_tokens=True)


def compute_pair_baselines(model, tokenizer, row, patch_position="cot_end"):
    """Build prompts + tokenize + unpatched hash logprobs for a (src, tgt)
    pair. The prompt shape depends on `patch_position`:

      cot_end       — prompt = `chat_template(question + INSTRUCTION) + donor_CoT`.
                      Last token is the readout right before ####; patch is
                      applied there and we measure log P(####) etc.
      cot_start     — prompt = `chat_template(question + INSTRUCTION)` (NO donor
                      CoT). Last token is "the model is about to begin writing
                      its own CoT". The patch is applied at this position and
                      then the model freely generates CoT + answer; recovery is
                      measured behaviorally.
      mid_cot_<X>   — Asymmetric. Donor prompt = filler-DF question +
                      first X tokens of the gsm_symbolic donor CoT
                      (source_cot_prefix). Recipient prompt = noop question
                      + first X tokens of noop's own natural CoT
                      (target_own_cot_prefix). The patch fires at position X
                      of each side; the recipient has been traversing its
                      broken plan up to X, and the donor has been traversing
                      the symbolic reference. After the patch the model
                      freely generates the rest. Sweeping X maps where in
                      noop's CoT generation the commit becomes too set to
                      rescue. Pairs where either CoT is shorter than X
                      tokens are skipped (denom set to -inf).

    Returns a dict the patching kernel reads; caller can short-circuit on
    `denom_source_minus_target` before patching/generation.
    """
    device = next(model.parameters()).device

    if patch_position == "cot_end":
        # Trim trailing whitespace from the donor CoT (the `sym_reasoning`
        # column in cot_swap CSVs has no consistent trailing newlines) and
        # re-append the canonical separator that immediately precedes `####`
        # in the original CoT. This puts the last token of the prompt at the
        # newline right before `####`, so log P(####) at the readout reflects
        # "model is about to emit ####" — matching the activation-patching
        # convention. Without this, the last token is e.g. a period and the
        # readout reflects "model is about to emit \n", which makes the
        # log-P(####) cliff appear muted (~−30 nats vs ~0).
        src_prefix = row["source_cot_prefix"].rstrip() + "\n\n"
        target_cot = row.get("target_cot_prefix", None)
        if not isinstance(target_cot, str) or not target_cot:
            target_cot = row["source_cot_prefix"]
        tgt_prefix = target_cot.rstrip() + "\n\n"
    elif patch_position == "cot_start":
        # No donor CoT — model must generate its own reasoning after the patch.
        src_prefix = ""
        tgt_prefix = ""
    elif patch_position.startswith("mid_cot_"):
        try:
            X = int(patch_position.split("_")[-1])
        except (ValueError, IndexError):
            raise ValueError(
                f"mid_cot patch_position must be 'mid_cot_<X>' with integer X; "
                f"got {patch_position!r}"
            )
        # Donor side mirrors cot_end: filler-DF question + the symbolic
        # gsm_symbolic donor CoT (source_cot_prefix), truncated to first X
        # tokens. Recipient side traverses its OWN broken noop CoT for X
        # tokens (target_own_cot_prefix), so the patch tests "can we rescue
        # at position X of noop's natural reasoning by injecting the donor's
        # commit-readiness from the symbolic trajectory."
        donor_cot = row.get("source_cot_prefix", "")
        recipient_cot = row.get("target_own_cot_prefix", "")
        if not (isinstance(recipient_cot, str) and recipient_cot):
            raise ValueError(
                "mid_cot requires target_own_cot_prefix in the pair-loader "
                "output (recipient's natural CoT). Add it via the loader "
                "updates from 2026-05-20."
            )
        src_prefix = _truncate_cot_strict(donor_cot, X, tokenizer)
        tgt_prefix = _truncate_cot_strict(recipient_cot, X, tokenizer)
        if src_prefix is None or tgt_prefix is None:
            # Caller checks `denom_source_minus_target == -inf` (via the gap
            # filter, with -inf < any threshold) and skips. We can't fully
            # build the prompts here without one side being shorter than
            # X tokens, which would collapse positions and pollute results.
            return {
                "src_in": None, "tgt_in": None,
                "src_last": None, "tgt_last": None,
                "src_lp": None, "tgt_lp": None,
                "denom": float("-inf"),
                "tok_hash": None,
                "patch_position": patch_position,
                "mid_cot_too_short": True,
            }
    elif patch_position == "anchor_clause":
        # Per-pair patch position at the divergence-DLA anchor token (start
        # of the wrong-plan engagement clause in noop's natural CoT). The
        # anchor token index is precomputed per-pair via find_anchor() and
        # injected as row["_anchor_idx"] by the run loop (see the
        # anchor_clause block in run_cot_swap_head_scaling.run()).
        # Donor side: no CoT prefix (source-side prompt is unused at this
        # position — same convention as cot_start; denom set to -inf below).
        anchor_idx = row.get("_anchor_idx")
        # _anchor_idx is set per-pair upstream; missing/NaN means the pair had
        # no detectable divergence anchor — the run loop should have dropped
        # these before getting here.
        if anchor_idx is None or (isinstance(anchor_idx, float) and anchor_idx != anchor_idx):
            raise ValueError(
                "anchor_clause patch_position requires a precomputed _anchor_idx "
                "column on the pair row. Set it via find_anchor() in the run loop."
            )
        anchor_idx = int(anchor_idx)
        recipient_cot = row.get("target_own_cot_prefix", "")
        if not (isinstance(recipient_cot, str) and recipient_cot):
            raise ValueError(
                "anchor_clause requires target_own_cot_prefix on the pair row."
            )
        tgt_prefix = _truncate_cot_strict(recipient_cot, anchor_idx, tokenizer)
        src_prefix = ""  # unused; matches cot_start convention
        if tgt_prefix is None:
            return {
                "src_in": None, "tgt_in": None,
                "src_last": None, "tgt_last": None,
                "src_lp": None, "tgt_lp": None,
                "denom": float("-inf"),
                "tok_hash": None,
                "patch_position": patch_position,
                "mid_cot_too_short": True,
            }
    else:
        raise ValueError(
            f"Unknown patch_position: {patch_position!r}. "
            "Use 'cot_end', 'cot_start', 'mid_cot_<X>' (integer X), "
            "or 'anchor_clause' (per-pair anchor from divergence DLA)."
        )

    src_prompt, _, _ = build_prompt_and_question_span(
        row["source_question"], tokenizer, answer_prefix=src_prefix
    )
    tgt_prompt, _, _ = build_prompt_and_question_span(
        row["target_question"], tokenizer, answer_prefix=tgt_prefix
    )
    src_in = encode_prompt(src_prompt, tokenizer, device)
    tgt_in = encode_prompt(tgt_prompt, tokenizer, device)

    tok_hash_ids = tokenizer.encode("####", add_special_tokens=False)
    if len(tok_hash_ids) != 1:
        raise ValueError(f"Expected single-token ####, got {tok_hash_ids}")
    tok_hash = tok_hash_ids[0]

    # Log P(####) at the readout token is only meaningful when the prompt
    # ends at the canonical "model is about to emit ####" position — i.e.
    # cot_end with the EOL-normalized donor CoT. At cot_start (no donor CoT)
    # or mid_cot_X (truncated own CoT mid-step), the last prompt token is
    # nowhere near a `####` emission, so src_lp ≈ tgt_lp ≈ noise and the
    # `denom = src_lp - tgt_lp` filter is uninformative. Skip the two
    # forward passes here to save compute and leave the hash fields None.
    if patch_position == "cot_end":
        def hash_lp(inputs):
            with torch.no_grad():
                out = model(**inputs)
                lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
            return lp[0, tok_hash].item()
        src_lp = hash_lp(src_in)
        tgt_lp = hash_lp(tgt_in)
        denom_value = src_lp - tgt_lp
    else:
        src_lp = None
        tgt_lp = None
        # Sentinel for "no log-P(####) denominator". Callers must skip the
        # min_source_target_gap comparison outside cot_end.
        denom_value = float("-inf")
    return {
        "src_in":         src_in,
        "tgt_in":         tgt_in,
        "src_last":       int(src_in["input_ids"].shape[1]) - 1,
        "tgt_last":       int(tgt_in["input_ids"].shape[1]) - 1,
        "src_lp":         src_lp,
        "tgt_lp":         tgt_lp,
        "denom":          denom_value,
        "tok_hash":       tok_hash,
        "patch_position": patch_position,
    }


def head_patch_one(model, tokenizer, row, head_cells, n_heads, head_dim,
                   direction="restore", measure_final_answer=True,
                   max_new_tokens=1024, baselines=None,
                   patch_position="cot_end"):
    """Patch specified per-head outputs at the chosen prefill position; return
    metrics.

    `patch_position`:
      cot_end  (default) — last token of the full prompt (just before ####).
                           Standard readout for log P(####)-based metrics.
      cot_start          — first token of the donor CoT prefix. Tests whether
                           the head-set encodes the answer-marker plan early
                           in the CoT span; usually paired with
                           --measure_final_answer to read out behavioral
                           recovery via generation.

    When `measure_final_answer=True`, also generate after the patched prefill
    and record (a) the model's emitted answer and (b) whether it matches the
    target's gold answer. The same generate() call's step-0 logits give
    log P(####), so only one forward path is used.

    Pass a precomputed `baselines` dict (from `compute_pair_baselines`) to
    skip the redundant source/target forward passes.
    """
    if baselines is None:
        baselines = compute_pair_baselines(model, tokenizer, row, patch_position)
    # Sanity: caller must compute baselines with the same patch_position so
    # the prompt shape matches what we patch. (cot_start baselines drop the
    # donor CoT from the prompt, so the same `src_in` can't be reused across
    # positions.)
    if baselines.get("patch_position", "cot_end") != patch_position:
        raise ValueError(
            f"compute_pair_baselines was called with "
            f"patch_position={baselines.get('patch_position')!r} but head_patch_one "
            f"is being invoked with patch_position={patch_position!r}; rebuild "
            "baselines with the matching position."
        )
    src_in   = baselines["src_in"]
    tgt_in   = baselines["tgt_in"]
    # Both modes patch at the LAST token of their respective prompts:
    #   cot_end: last token of [question + chat_template + donor_CoT] → readout.
    #   cot_start: last token of [question + chat_template]            → just
    #              before the model starts generating its own CoT.
    src_pos = baselines["src_last"]
    tgt_pos = baselines["tgt_last"]
    src_lp   = baselines["src_lp"]
    tgt_lp   = baselines["tgt_lp"]
    denom    = baselines["denom"]
    tok_hash = baselines["tok_hash"]

    def hash_lp(inputs):
        with torch.no_grad():
            out = model(**inputs)
            lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
        return lp[0, tok_hash].item()

    if direction == "restore":
        donor_in, recv_in = src_in, tgt_in
        donor_last, recv_last = src_pos, tgt_pos
        baseline_lp, other_lp = tgt_lp, src_lp
    else:
        donor_in, recv_in = tgt_in, src_in
        donor_last, recv_last = tgt_pos, src_pos
        baseline_lp, other_lp = src_lp, tgt_lp
    # `norm_denom` is the unpatched src/tgt gap on log P(####), which only
    # makes sense at cot_end. At cot_start / mid_cot we skip the baseline
    # forward passes entirely, so src_lp and tgt_lp are None and the gap
    # quantity is undefined; downstream `normalized_effect` and
    # `readout_delta` will also be None.
    if patch_position == "cot_end" and baseline_lp is not None and other_lp is not None:
        norm_denom = other_lp - baseline_lp
    else:
        norm_denom = None

    # 1) Capture donor per-head outputs at the readout position.
    cells_by_bi = {}
    for L, h in head_cells:
        cells_by_bi.setdefault(L - 1, []).append(h)
    block_idxs = sorted(cells_by_bi)

    captured = {}

    def make_capture(bi, last_pos):
        def hook(_m, inputs):
            x = inputs[0]
            B, seq, hidden = x.shape
            # Autoregressive guard: only capture during prefill where the
            # readout position exists in this chunk.
            if seq <= last_pos:
                return None
            x_view = x.view(B, seq, n_heads, head_dim)
            captured[bi] = x_view[:, last_pos, :, :].detach().clone()
        return hook

    handles = []
    try:
        for bi in block_idxs:
            m = model.model.layers[bi].self_attn.o_proj
            handles.append(m.register_forward_pre_hook(make_capture(bi, donor_last)))
        with torch.no_grad():
            model(**donor_in)
    finally:
        for h in handles:
            h.remove()

    # 2) Recipient forward with per-head patch hooks at the readout position.
    def make_patch(bi, heads, last_pos):
        donor_slice = captured[bi]

        def hook(_m, inputs):
            x = inputs[0].clone()
            B, seq, hidden = x.shape
            # Autoregressive guard: only patch during prefill. Generation
            # steps see seq=1 (with KV-cache), where last_pos is out of range.
            if seq <= last_pos:
                return None
            x_view = x.view(B, seq, n_heads, head_dim)
            for h in heads:
                x_view[:, last_pos, h, :] = donor_slice[:, h, :].to(
                    device=x.device, dtype=x.dtype
                )
            return (x_view.view(B, seq, hidden),)
        return hook

    handles = []
    extra = {}
    try:
        for bi, heads in cells_by_bi.items():
            m = model.model.layers[bi].self_attn.o_proj
            handles.append(m.register_forward_pre_hook(make_patch(bi, heads, recv_last)))
        if measure_final_answer:
            gold = row.get("target_answer") if direction == "restore" \
                else row.get("source_answer")
            gen = _generate_and_score(
                model, tokenizer, recv_in, gold_answer=gold,
                tok_hash=tok_hash, max_new_tokens=max_new_tokens,
            )
            patched_lp = gen["patched_hash_logprob"]
            extra = {
                "gold_answer":                    str(gold) if gold is not None else None,
                "patched_hash_emitted":           gen["patched_hash_emitted"],
                "patched_final_answer_success":   gen["patched_final_answer_success"],
                "patched_gold_logprob":           gen["patched_gold_logprob"],
                "patched_gold_immediate":         gen["patched_gold_immediate"],
                "patched_emitted_answer":         gen["patched_emitted_answer"],
                "patched_generation":             gen["patched_generation"],
            }
        else:
            patched_lp = hash_lp(recv_in) if patch_position == "cot_end" else None
    finally:
        for h in handles:
            h.remove()

    # `readout_delta` and `normalized_effect` are only meaningful when we ran
    # the baseline hash_lp (cot_end). For cot_start / mid_cot they're None.
    if baseline_lp is not None and patched_lp is not None:
        readout_delta = patched_lp - baseline_lp
        normalized_effect = (
            readout_delta / norm_denom
            if norm_denom is not None and abs(norm_denom) > 1e-6 else None
        )
    else:
        readout_delta = None
        normalized_effect = None

    return {
        "source_hash_logprob":         src_lp,
        "target_hash_logprob":         tgt_lp,
        "denom_source_minus_target":   denom,
        "direction":                   direction,
        "patch_position":              patch_position,
        "baseline_hash_logprob":       baseline_lp,
        "norm_denom":                  norm_denom,
        "patched_hash_logprob":        patched_lp,
        "readout_delta":               readout_delta,
        "normalized_effect":           normalized_effect,
        "num_cells":                   int(len(head_cells)),
        **extra,
    }


def _cells_signature(head_cells):
    """Short stable hash of a sorted cell list; stamped on every JSONL row."""
    canonical = ",".join(f"{L}:{h}" for L, h in sorted(head_cells))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def run(model, tokenizer, model_name, contrast, head_cells, label,
        direction="restore", min_source_target_gap=0.5, overwrite=False,
        measure_final_answer=True, max_new_tokens=1024,
        pair_source=None, cot_source=None, patch_position="cot_end",
        filter_flipped=False):
    """`contrast` is used for the DLA cell-set lookup and the output directory.
    `pair_source` (defaults to `contrast`) is the cot_swap CSV the pair data
    comes from. `cot_source` (defaults to `pair_source`) is the contrast from
    which `source_cot_prefix` is taken — pass a different value to keep
    `pair_source`'s questions (for length control) but swap in another
    contrast's donor CoT (e.g., `cot_source=noop_clean` for symbolic's
    clean CoT).

    `patch_position` (default cot_end) selects the prefill slot where the
    per-head V outputs are captured/patched. The output dir is tagged with
    the position so cot_start and cot_end runs at the same label coexist.

    `filter_flipped` (default False): drop pairs where noop's natural
    re-generation produces the gold answer (despite being labeled
    noop_clean_wrong by the original inference pipeline). Output dir gets
    a `_nofl` suffix so filtered/unfiltered runs don't collide.
    """
    if pair_source is None:
        pair_source = contrast
    if cot_source is None:
        cot_source = pair_source
    # Tag the output dir with patch_position so cot_start results, EOL-fixed
    # cot_end results, and the legacy (no-EOL) cot_end results all live in
    # separate directories. The legacy "" tag is reserved for pre-2026-05-20
    # cot_end runs whose donor CoT ended with a period (last token of prompt
    # was "."); today's cot_end now appends "\n\n" so the readout falls on
    # the newline right before ####. New cot_end output goes to `_cot_end_eol`.
    if patch_position == "cot_end":
        pos_tag = "_cot_end_eol"
    else:
        pos_tag = f"_{patch_position}"
    if filter_flipped:
        pos_tag += "_nofl"
    out_dir = OUT_DIR / model_name / contrast / f"{label}{pos_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{direction}.jsonl"
    cells_path = out_dir / "head_cells.json"
    cells_sig = _cells_signature(head_cells)

    # Reject mismatched-cells reuse before clobbering metadata or appending
    # to a JSONL computed against a different cell set.
    if cells_path.exists():
        prev = json.load(open(cells_path))
        prev_cells = [tuple(c) for c in prev.get("head_cells", [])]
        prev_sig = prev.get("cells_signature") or _cells_signature(prev_cells)
        if prev_sig != cells_sig:
            if out_path.exists() and not overwrite:
                raise SystemExit(
                    f"head_cells mismatch under label '{label}':\n"
                    f"  existing  ({prev_sig}): {prev_cells[:5]}... "
                    f"(n={len(prev_cells)})\n"
                    f"  new       ({cells_sig}): {head_cells[:5]}... "
                    f"(n={len(head_cells)})\n"
                    f"Pass --overwrite to clear results, or use a different --label."
                )

    if overwrite and out_path.exists():
        out_path.unlink()
    cells_path.write_text(json.dumps({
        "head_cells":        head_cells,
        "num_cells":         len(head_cells),
        "cells_signature":   cells_sig,
    }, indent=2))

    # Resume only matches rows from the same cell signature, so a mid-run
    # cell-set change can never silently mix provenance. When
    # `measure_final_answer` is on, also require the new field to be present
    # — rows without it are physically dropped from the file (to avoid
    # duplicates after recomputation) and re-run.
    done = set()
    if out_path.exists():
        n_match = n_mismatch = n_upgrade = 0
        keep_lines = []
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("cells_signature") != cells_sig:
                    n_mismatch += 1
                    continue
                if measure_final_answer and "patched_final_answer_success" not in r:
                    n_upgrade += 1
                    continue
                keep_lines.append(line if line.endswith("\n") else line + "\n")
                done.add((int(r["original_id"]), int(r["instance"])))
                n_match += 1
        if n_mismatch:
            raise SystemExit(
                f"{out_path} contains {n_mismatch} rows with cells_signature "
                f"different from current set ({cells_sig}). Pass --overwrite "
                f"to clear, or pick a different --label."
            )
        if n_upgrade:
            out_path.write_text("".join(keep_lines))
            print(f"Resuming — {n_match} rows already done; "
                  f"{n_upgrade} rows lacked final-answer fields and were "
                  f"dropped (will be recomputed).")
        else:
            print(f"Resuming — {n_match} rows already done")

    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_heads

    df = load_cot_swap_as_patching_pairs(source=pair_source, cot_source=cot_source)
    if filter_flipped:
        before = len(df)
        df = filter_flipped_pairs(df)
        print(f"filter_flipped: dropped {before - len(df)} flipped pairs → {len(df)}")
    cot_note = f"; CoT from `{cot_source}`" if cot_source != pair_source else ""
    print(f"Loaded {len(df)} pairs from `{pair_source}`{cot_note}; "
          f"{len(head_cells)} cells (DLA from `{contrast}`); direction={direction}")
    model.eval()

    skipped_gap = 0
    skipped_too_short = 0
    with open(out_path, "a") as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=label):
            key = (int(row["original_id"]), int(row["instance"]))
            if key in done:
                continue
            # Pre-compute the (src, tgt) baselines once per row; if the
            # source-target hash-logprob gap is below the filter threshold,
            # skip without doing the patched forward / generation.
            # Note: cot_start mode builds prompts WITHOUT the donor CoT, so
            # the "gap" here is log P(####) on plain question prompts —
            # almost always near-zero. Use a low/zero gap filter for
            # cot_start by passing --min_source_target_gap 0 from the CLI;
            # the default of 0.5 will drop most cot_start rows.
            baselines = compute_pair_baselines(
                model, tokenizer, row, patch_position=patch_position
            )
            if baselines.get("mid_cot_too_short"):
                skipped_too_short += 1
                continue
            # Gap-filter only applies to cot_end where log P(####) is the
            # readout. Non-cot_end positions skip the baseline forward
            # passes (denom is sentinel -inf for those), so we don't compare.
            if patch_position == "cot_end" and baselines["denom"] < min_source_target_gap:
                skipped_gap += 1
                continue
            metrics = head_patch_one(
                model, tokenizer, row, head_cells,
                n_heads=n_heads, head_dim=head_dim, direction=direction,
                measure_final_answer=measure_final_answer,
                max_new_tokens=max_new_tokens,
                baselines=baselines,
                patch_position=patch_position,
            )
            rec = {
                "original_id":      int(row["original_id"]),
                "instance":         int(row["instance"]),
                "contrast":         contrast,
                "pair_source":      pair_source,
                "cot_source":       cot_source,
                "label":            label,
                "cells_signature":  cells_sig,
                "source_label":     row["source_label"],
                "target_label":     row["target_label"],
                **metrics,
            }
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()

    if skipped_too_short:
        print(f"Skipped {skipped_too_short} rows with one or both CoTs shorter than X tokens (mid_cot)")
    if skipped_gap:
        print(f"Skipped {skipped_gap} rows with source-target gap < {min_source_target_gap}")
    print(f"Results → {out_path}")


def plot(model_name, contrast, label, direction="restore",
         min_source_target_gap=0.5, layer_overlay_path=None,
         patch_position="cot_end", filter_flipped=False):
    """Single bar: head-cell patch normalized effect, with bootstrap CI.

    If `layer_overlay_path` points to the corresponding cot_swap layer-scope
    restore.jsonl, overlay its per-layer normalized effects (the Figure-6 cliff)
    so the head-cell bar sits next to the residual-stream ceiling.
    """
    if patch_position == "cot_end":
        pos_tag = "_cot_end_eol"
    else:
        pos_tag = f"_{patch_position}"
    if filter_flipped:
        pos_tag += "_nofl"
    out_dir = OUT_DIR / model_name / contrast / f"{label}{pos_tag}"
    # The built-in plotter shows normalized log P(####) recovery — only
    # meaningful at cot_end where log P(####) is the actual readout. For
    # cot_start / mid_cot_X, rows have normalized_effect = None by design
    # (no #### baseline was computed). Behavioral plots come from
    # plot_recovery_sweep.py instead.
    if patch_position != "cot_end":
        print(f"Skipping built-in log-P(####) plot for patch_position={patch_position!r}.")
        print(f"  Use: python interpretability/plot_recovery_sweep.py")
        print(f"  for behavioral metrics on this output dir: {out_dir}")
        return
    path = out_dir / f"{direction}.jsonl"
    if not path.exists():
        print(f"No results at {path}")
        return
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for r in rows
            if r["normalized_effect"] is not None
            and float(r["denom_source_minus_target"]) >= min_source_target_gap]
    if not rows:
        print("No usable rows after gap filter.")
        return
    effects = np.array([r["normalized_effect"] for r in rows])
    cluster = np.array([r["original_id"] for r in rows])

    uniq, inv = np.unique(cluster, return_inverse=True)
    per_template = np.zeros(len(uniq))
    for i in range(len(uniq)):
        per_template[i] = effects[inv == i].mean()
    mean = float(per_template.mean())
    rng = np.random.default_rng(0)
    n_boot = 2000
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(per_template), size=len(per_template))
        boots[b] = per_template[idx].mean()
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([f"head-cell patch\n({label})"], [mean],
           yerr=[[mean - lo], [hi - mean]], capsize=6, color="#2166ac", alpha=0.9)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5, label="no recovery")
    ax.axhline(1, color="black", lw=0.8, alpha=0.3, ls="--", label="full recovery (residual ceiling)")
    ax.set_ylabel("Normalized patching effect")
    ax.set_title(
        f"{contrast}\n{direction}: head-cell patch  |  "
        f"N={len(rows)} rows, {len(uniq)} templates",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = out_dir / f"{direction}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {out}  (mean={mean:.3f}, 95%CI=[{lo:.3f}, {hi:.3f}])")


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
                        help="Pair-data source for the (source_question, "
                             "target_question) pair. Defaults to --contrast.")
    parser.add_argument("--cot_source", default=None,
                        choices=list(COT_SWAP_PATCHING_CONTRASTS),
                        help="Source of the donor CoT (`source_cot_prefix`). "
                             "Defaults to --pair_source. Pass a different "
                             "value to combine pair_source's questions (for "
                             "length control) with another contrast's CoT — "
                             "e.g., `--pair_source filler_df_correct_vs_noop_clean_wrong "
                             "--cot_source noop_clean` for length-matched "
                             "filler-DF/noop questions but with symbolic's "
                             "clean CoT as the donor reasoning.")
    cells = parser.add_mutually_exclusive_group(required=False)
    cells.add_argument("--cells", type=str,
                       help="Space- or comma-separated 'L:h' cells "
                            "(paper layers).")
    cells.add_argument("--cells_from_dla", action="store_true",
                       help="Auto-pick wrong-direction cells from the cot_swap "
                            "DLA results for the same contrast.")
    parser.add_argument("--dla_threshold", type=float, default=1e-4,
                        help="|dla_mean| threshold for cells_from_dla.")
    parser.add_argument("--dla_mode", default="wrong_direction",
                        choices=list(DLA_MODES),
                        help="Cell-selection recipe when --cells_from_dla:\n"
                             "  wrong_direction:    noop dla < -threshold (anti-####)\n"
                             "  filler_supporting:  filler dla > +threshold (pro-#### on filler)\n"
                             "  pro_noop:           noop dla > +threshold (pro-#### on noop)\n"
                             "  combined:           union of wrong_direction + filler_supporting")
    parser.add_argument("--min_noop_attn", type=float, default=0.0,
                        help="Additional filter: only keep cells whose noop-side "
                             "clause attention (attn_mean) is at least this value. "
                             "Use to restrict to cells where the clause path is a "
                             "big enough fraction of the head's readout output that "
                             "swapping it dominates over non-clause writes.")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=list(range(22, 37)),
                        help="Layers to scan when --cells_from_dla.")
    parser.add_argument("--direction", choices=["restore", "disrupt"],
                        default="restore")
    parser.add_argument("--min_source_target_gap", type=float, default=0.5)
    parser.add_argument("--label", type=str, default=None,
                        help="Sub-directory name for this cell set.")
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
                        help="Prefill slot at which to capture/patch per-head "
                             "V. Options:\n"
                             "  cot_end       — last token before #### (donor "
                             "CoT in prompt; log P(####)-style metrics).\n"
                             "  cot_start     — last token of no-donor-CoT "
                             "prompt (=  assistant header end). Pair with "
                             "--measure_final_answer.\n"
                             "  mid_cot_<X>   — feed first X tokens of EACH "
                             "side's OWN CoT, then patch and let the model "
                             "freely continue. Sweep X to find when the cliff "
                             "activates during CoT generation.")
    parser.add_argument("--filter_flipped", action="store_true",
                        help="Drop pairs where noop's natural re-generation "
                             "produces the gold answer (despite being labeled "
                             "noop_clean_wrong). Output dir gets `_nofl` "
                             "suffix.")
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]

    if args.plot_only:
        if not args.label:
            raise SystemExit("--plot_only requires --label to pick the result dir.")
        plot(model_name, args.contrast, args.label, args.direction,
             min_source_target_gap=args.min_source_target_gap,
             patch_position=args.patch_position,
             filter_flipped=args.filter_flipped)
        raise SystemExit(0)

    if args.cells_from_dla:
        head_cells = cells_from_dla(
            model_name, args.contrast, args.layers,
            threshold=args.dla_threshold, mode=args.dla_mode,
            min_noop_attn=args.min_noop_attn,
        )
        if not head_cells:
            raise SystemExit(
                f"No cells found for mode={args.dla_mode} with "
                f"threshold={args.dla_threshold} and "
                f"min_noop_attn={args.min_noop_attn}; "
                f"loosen one of those thresholds."
            )
        print(f"Selected {len(head_cells)} cells from DLA "
              f"(mode={args.dla_mode}, dla>{args.dla_threshold}, "
              f"noop_attn>={args.min_noop_attn}).")
    elif args.cells:
        head_cells = parse_cells(args.cells)
    else:
        raise SystemExit("Pass --cells or --cells_from_dla.")

    label = _label_for_cells(args.label, args.cells_from_dla,
                             args.dla_threshold, args.dla_mode,
                             min_noop_attn=args.min_noop_attn)
    # When pair_source or cot_source differs from the contrast, tag the
    # output label so the two experiments don't clobber each other.
    pair_source = args.pair_source or args.contrast
    cot_source = args.cot_source or pair_source
    if pair_source != args.contrast:
        label = f"{label}_pair-{pair_source}"
    if cot_source != pair_source:
        label = f"{label}_cot-{cot_source}"
    print(f"Cell set label: {label}")
    print(f"First 10 cells: {head_cells[:10]}")
    print(f"DLA contrast (output + cell lookup): {args.contrast}")
    print(f"Pair source (questions):             {pair_source}")
    print(f"CoT source (donor reasoning):        {cot_source}")

    model, tokenizer = load_model(args.model_id)
    run(model, tokenizer, model_name, args.contrast, head_cells, label,
        direction=args.direction,
        min_source_target_gap=args.min_source_target_gap,
        overwrite=args.overwrite,
        measure_final_answer=args.measure_final_answer,
        max_new_tokens=args.max_new_tokens,
        pair_source=pair_source,
        cot_source=cot_source,
        patch_position=args.patch_position,
        filter_flipped=args.filter_flipped)
    plot(model_name, args.contrast, label, args.direction,
         min_source_target_gap=args.min_source_target_gap,
         patch_position=args.patch_position,
         filter_flipped=args.filter_flipped)
