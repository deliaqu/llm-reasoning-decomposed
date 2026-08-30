"""
Activation patching for CoT-swap readiness.

For P1 vs padded-symbolic examples, append the padded-symbolic CoT prefix to
both prompts, patch activations from the padded-symbolic question representation
into the P1 question representation, and measure log P(next token = ####) at the
end of the shared CoT prefix.

This complements run_cot_swap_logit_lens.py: the logit lens asks when ####
readiness is linearly visible; this script asks whether patched question
activations causally contribute to that readiness under a fixed downstream CoT
trace.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ACTIVATION_PATCHING_RESULT_DIR, COT_INSTRUCTION, MODEL_NAME_MAP
from run_cot_swap_logit_lens import (
    _bootstrap_ci,
    _load_p1_contrast_data,
    _normalize_subsets,
    _style_layer_axis,
    _subset_tag,
    _template_means,
)
from utils.noop_utils import load_cot_swap_as_patching_pairs, load_model, to_serializable
from utils.shared_utils import remove_all_hooks, untuple

COT_SWAP_PATCHING_CONTRASTS = {
    "filler_vs_noop_clean",
    "filler_df_clean_vs_noop_clean_fail",
    "filler_df_correct_vs_noop_clean_wrong",
    "filler_clean_vs_noop_clean_fail",
    "filler_correct_vs_noop_clean_wrong",
    "noop_clean",           # source=symbolic, target=noop_clean — donor's CoT
                            # is the clean symbolic reasoning (no inserted clause)
    "noop_clean_within_template",
    "noop_clean_within_template_clean_vs_fail",
    # Self-paired noop_clean_wrong (source == target == noop_clean question).
    # No donor CoT, no filler-side trajectory. Only meaningful at cot_start
    # with head_scaling; the loader is in noop_utils.load_cot_swap_as_patching_pairs.
    "noop_clean_wrong_self",
    # noop_clean_wrong with set-diff via sym_question (not filler-DF). Larger
    # population than filler_df_correct_vs_noop_clean_wrong.
    "noop_clean_wrong_sym_diff",
    # Self-paired GSM-Symbolic CoT-correct (clean-baseline head_scaling).
    "symbolic_correct_self",
    # Self-paired GSM-Filler (digit-free) CoT-correct (clean-baseline
    # damage runs at full filler population, ~4700 pairs / 100 templates).
    "filler_correct_self",
    # P1 (extra-step) vs padded-Symbolic (length-matched). Used for the
    # symbolic-formulation divergence DLA / head_scaling experiments.
    "p1_correct_vs_padded",
}


OUT_DIR = Path(ACTIVATION_PATCHING_RESULT_DIR).parent / "cot_swap_activation_patching"

PATCHING_STAGE_BOUNDARIES = [
    (22, "Abstraction"),
    (36, "Symbolic\nformulation"),
]


def output_path(model_name, contrast, padded_dataset, p1_value_subset, scope,
                direction="restore", patch_positions="question_end",
                allow_truncated_span=False, layers=None, length_filter="all",
                cot_source=None):
    """Return the result path.

    Layout:
        {model}/{contrast}/{padded_dataset}/{length_filter}/{subset_dir}/{patch_positions}/{scope}/{direction}{trunc}{layers}.jsonl

    For cot_swap patching contrasts (no padded_dataset / p1_value_subset),
    the `padded_dataset`, `length_filter`, and `subset_dir` levels are omitted.

    `cot_source` (defaults to contrast): when set to a different cot_swap
    contrast, the contrast directory gets a `_cot-{cot_source}` suffix so
    hybrid runs don't clobber the canonical contrast directory.
    """
    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    trunc_tag = "_truncated" if allow_truncated_span and patch_positions == "question_span" else ""
    if contrast in COT_SWAP_PATCHING_CONTRASTS:
        contrast_dir = contrast
        if cot_source is not None and cot_source != contrast:
            contrast_dir = f"{contrast}_cot-{cot_source}"
        return (
            OUT_DIR / model_name / contrast_dir
            / patch_positions / scope / f"{direction}{trunc_tag}{layers_tag}.jsonl"
        )
    subsets = _normalize_subsets(p1_value_subset)
    subset_dir = "all" if subsets == ["all"] else "+".join(subsets)
    # `_cot-symbolic_aligned` marks the post-2026-05-20 convention where the
    # donor CoT comes from gsm_symbolic_to_p1 (the un-padded original symbolic
    # with the same value instantiation) rather than from padded_symbolic's
    # own regenerated CoT. Tag is permanent so old (pre-fix) runs at the
    # untagged path stay untouched.
    contrast_dir = f"{contrast}_cot-symbolic_aligned"
    return (
        OUT_DIR / model_name / contrast_dir / padded_dataset / length_filter / subset_dir
        / patch_positions / scope / f"{direction}{trunc_tag}{layers_tag}.jsonl"
    )


def _add_shard_suffix(path, shard):
    """Append `.shard_{k}_of_{N}` before the suffix when sharded; else
    return `path` unchanged. Used so layer-shard writes coexist in the
    same directory and the consumer-side glob can stitch them back.
    """
    if shard is None:
        return path
    k, n = shard
    return path.with_name(f"{path.stem}.shard_{k}_of_{n}{path.suffix}")


def baselines_path_for_cot_swap(model_name, contrast, cot_source=None,
                                padded_dataset=None, p1_value_subset=None,
                                length_filter="all"):
    """Path to the precomputed gold-logprob baseline cache.

    For cot_swap contrasts the layout is keyed on `(contrast, cot_source)`.
    For `p1_vs_padded_symbolic` we also need `padded_dataset`,
    `length_filter`, and the p1 value subset because those control which
    pairs the loader emits — different padded datasets give different
    (source, target) prompts so their baselines aren't interchangeable.

    Baselines are direction-agnostic (the same unpatched source/target
    generations feed both restore and disrupt), so we cache one per
    distinct prompt set.
    """
    if contrast == "p1_vs_padded_symbolic":
        subsets = _normalize_subsets(p1_value_subset)
        subset_dir = "all" if subsets == ["all"] else "+".join(subsets)
        # See output_path() for the `_cot-symbolic_aligned` rationale.
        contrast_dir = f"{contrast}_cot-symbolic_aligned"
        return (
            OUT_DIR / model_name / contrast_dir / padded_dataset
            / length_filter / subset_dir / "baselines.jsonl"
        )
    contrast_dir = contrast
    if cot_source is not None and cot_source != contrast:
        contrast_dir = f"{contrast}_cot-{cot_source}"
    return OUT_DIR / model_name / contrast_dir / "baselines.jsonl"


def load_cot_swap_baselines(path):
    """Read a baselines.jsonl into a dict keyed by (original_id, instance).
    Missing file → empty dict. Each row is parseable JSON with at least
    `original_id`, `instance`, `unpatched_source_gold_logprob`,
    `unpatched_target_gold_logprob`.

    Also picks up sharded outputs (`baselines.shard_k_of_N.jsonl`) written
    by data-parallel baseline jobs — keeps the consumer side unchanged.
    """
    cache = {}
    files = []
    if path.exists():
        files.append(path)
    files.extend(sorted(path.parent.glob(
        f"{path.stem}.shard_*_of_*{path.suffix}"
    )))
    for fp in files:
        for line in open(fp):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (int(r["original_id"]), int(r["instance"]))
            cache[key] = r
    return cache


def build_prompt_and_question_span(question, tokenizer, answer_prefix="",
                                   instruction=COT_INSTRUCTION):
    """Return prompt plus token span covering the question text."""
    messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
    prompt_no_prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_prompt = prompt_no_prefix + answer_prefix

    # Locate the end of the question inside the serialized chat prompt. If the
    # question string occurs multiple times, the first occurrence is the user
    # message body created above.
    char_start = prompt_no_prefix.index(str(question))
    char_end = char_start + len(str(question))

    # Prefer offset-mapping on the full prompt so we get the index that actually
    # appears at inference time, regardless of BPE merges across the question
    # boundary. Fall back to prefix-retokenization if the tokenizer is slow.
    try:
        enc = tokenizer(
            full_prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        offsets = enc["offset_mapping"]
        start_candidates = [
            i for i, (s, e) in enumerate(offsets) if s <= char_start < e
        ]
        if start_candidates:
            question_start_idx = start_candidates[0]
        else:
            question_start_idx = min(
                i for i, (s, e) in enumerate(offsets) if s >= char_start and e > s
            )
        candidates = [i for i, (s, e) in enumerate(offsets) if s < char_end <= e]
        if candidates:
            question_end_idx = candidates[0]
        else:
            question_end_idx = max(
                i for i, (s, e) in enumerate(offsets) if e <= char_end and e > s
            )
    except (TypeError, NotImplementedError):
        prefix_to_question_start = prompt_no_prefix[:char_start]
        prefix_to_question_end = prompt_no_prefix[:char_end]
        question_start_idx = len(
            tokenizer(prefix_to_question_start, add_special_tokens=False)["input_ids"]
        )
        question_end_idx = (
            len(tokenizer(prefix_to_question_end, add_special_tokens=False)["input_ids"]) - 1
        )
    return full_prompt, question_start_idx, question_end_idx


def encode_prompt(prompt, tokenizer, device):
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)


def make_save_hook(state_dict, key):
    def hook(_module, _inputs, output):
        state_dict[key] = untuple(output).detach().clone()

    return hook


def make_cross_position_patching_hook(source_state, source_idx, target_idx):
    """Forward hook that overwrites the receiver's output at `target_idx`
    positions with the donor's saved state at `source_idx`.

    Autoregressive guard: during `model.generate()`, the post-prefill steps
    feed only the newly-decoded token (output has shape (B, 1, hidden) with
    KV-cache). The target indices point into the original prompt and are
    out of range in those chunks, so we skip patching and pass the chunk
    through unchanged.
    """
    max_target_idx = max(target_idx) if target_idx else 0

    def hook(_module, _inputs, output):
        original_output = output
        output_tensor = untuple(output)
        if output_tensor.shape[1] <= max_target_idx:
            return original_output
        output_tensor = output_tensor.clone()
        source_tensor = untuple(source_state).to(output_tensor.device)
        output_tensor[:, target_idx, :] = source_tensor[:, source_idx, :]
        if isinstance(original_output, tuple):
            return (output_tensor,) + original_output[1:]
        return output_tensor

    return hook


def patch_indices(source_start, source_end, target_start, target_end,
                  patch_positions, allow_truncated_span=False,
                  source_input_len=None, target_input_len=None):
    """Return aligned source and target token indices for the requested site."""
    if patch_positions == "question_end":
        return [source_end], [target_end]
    if patch_positions == "question_span":
        source_span = source_end - source_start + 1
        target_span = target_end - target_start + 1
        if source_span != target_span and not allow_truncated_span:
            return [], []
        span_len = min(source_span, target_span)
        return (
            list(range(source_start, source_start + span_len)),
            list(range(target_start, target_start + span_len)),
        )
    if patch_positions == "cot_end":
        if source_input_len is None or target_input_len is None:
            raise ValueError("cot_end requires source_input_len and target_input_len")
        return [source_input_len - 1], [target_input_len - 1]
    if patch_positions == "cot_start":
        # Same kernel as cot_end (single-token patch at the last position),
        # but the prompt is built WITHOUT the donor CoT — see the
        # source_answer_prefix branch in `patch_one`. So source_input_len-1
        # here is the index of the assistant-header-end token, the slot
        # where the model would generate its first CoT token.
        if source_input_len is None or target_input_len is None:
            raise ValueError("cot_start requires source_input_len and target_input_len")
        return [source_input_len - 1], [target_input_len - 1]
    raise ValueError(f"Unknown patch_positions: {patch_positions}")


def layer_module(model, scope, layer_idx):
    if scope == "layer":
        return model.model.layers[layer_idx], f"layer_{layer_idx}"
    if scope == "mlp":
        return model.model.layers[layer_idx].mlp, f"layer_mlp_{layer_idx}"
    if scope == "attn_output":
        return model.model.layers[layer_idx].self_attn, f"layer_attn_output_{layer_idx}"
    raise ValueError(f"Unsupported scope for this runner: {scope}")


def hash_logprob(model, inputs, tok_hash):
    with torch.no_grad():
        out = model(**inputs)
        lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    return lp[0, tok_hash].item()


def generate_and_score(model, tokenizer, recv_inputs, gold_answer, tok_hash,
                       max_new_tokens=1024):
    """Run one greedy `model.generate()` (under whatever hooks are active) and
    return:

      patched_hash_logprob          — log P(next token = ####) at the readout
                                      (= step-0 next-token distribution).
      patched_hash_emitted          — bool: was `####` the FIRST generated
                                      token (immediate commit)?
      patched_final_answer_success  — bool (LOOSE): does the last number after
                                      the last `####` equal gold? Saturates
                                      at ~1.0 in practice.
      patched_gold_logprob          — log P(gold-answer first token) at the
                                      position right after the generated
                                      `####` boundary, recovered from the
                                      generation scores.
      patched_gold_immediate        — bool (TIGHT): at the post-`####` slot,
                                      did the model emit one of the gold
                                      first-token candidates? Discrete
                                      complement of patched_gold_logprob.

    Stops at the answer boundary (the first numeric token after ####) via
    `AnswerBoundaryStoppingCriteria`, keeping each call short and pinning the
    scorer to a stable token slot.
    """
    from transformers import StoppingCriteriaList
    from run_noop_activation_patching import (
        AnswerBoundaryStoppingCriteria,
        _extract_emitted_answer,
        _is_correct_final_answer,
        _gold_answer_logprob_from_generation_scores,
        _gold_first_token_candidates,
        _boundary_end_index,
    )

    prompt_len = recv_inputs["input_ids"].shape[1]
    prompt_ids = recv_inputs["input_ids"][0].detach().cpu().tolist()
    stopping = StoppingCriteriaList([
        AnswerBoundaryStoppingCriteria(tokenizer, prompt_len)
    ])
    with torch.no_grad():
        out = model.generate(
            **recv_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping,
            return_dict_in_generate=True,
            output_scores=True,
        )
    step0 = out.scores[0]
    lp = F.log_softmax(step0.float(), dim=-1)
    hash_lp = float(lp[0, tok_hash].item())
    gen_ids_tensor = out.sequences[0, prompt_len:]
    generation = tokenizer.decode(gen_ids_tensor, skip_special_tokens=True)
    emitted = _extract_emitted_answer(generation)
    success = (
        gold_answer is not None
        and _is_correct_final_answer(generation, str(gold_answer))
    )
    # Binary commit indicator: did the model emit `####` as its FIRST
    # generated token (immediate commit)? The looser "#### appears anywhere"
    # check saturated at ~1.0 because AnswerBoundaryStoppingCriteria stops at
    # #### and the model almost always emits it within max_new_tokens. The
    # argmax-at-step-0 version is the discrete complement of patched_hash_logprob
    # — a patch can lift log P(####) without the argmax ever choosing it, or
    # vice versa.
    hash_emitted = (
        gen_ids_tensor.numel() > 0
        and int(gen_ids_tensor[0].item()) == tok_hash
    )
    gold_lp = float("nan")
    gold_immediate = False
    if gold_answer is not None:
        cont_ids_list = gen_ids_tensor.detach().cpu().tolist()
        gold_lp = _gold_answer_logprob_from_generation_scores(
            model, tokenizer, prompt_ids,
            cont_ids_list, str(gold_answer), out.scores,
        )
        # Tight binary recovery: at the post-#### slot, did the model actually
        # generate one of the gold first-token candidates? Since decoding is
        # greedy (do_sample=False), the emitted token IS the argmax — so this
        # is the discrete complement of `patched_gold_logprob`. Complements the
        # loose `patched_final_answer_success` which only checks "is the last
        # number in the post-#### text equal to gold" and saturates at ~1.0.
        boundary_end = _boundary_end_index(tokenizer, cont_ids_list)
        if boundary_end is not None and boundary_end < len(cont_ids_list):
            gold_candidates = set(
                _gold_first_token_candidates(tokenizer, str(gold_answer))
            )
            gold_immediate = int(cont_ids_list[boundary_end]) in gold_candidates
    return {
        "patched_hash_logprob":         hash_lp,
        "patched_hash_emitted":         bool(hash_emitted),
        "patched_final_answer_success": bool(success),
        "patched_emitted_answer":       emitted,
        "patched_generation":           generation,
        "patched_gold_logprob":         float(gold_lp),
        "patched_gold_immediate":       bool(gold_immediate),
    }


def patch_one(model, tokenizer, row, scope, layers, min_source_target_gap,
              direction, patch_positions, allow_truncated_span,
              measure_final_answer=False, max_new_tokens=1024,
              cached_baseline=None):
    """Patch source/target question activations and measure P(next token = ####).

    When `measure_final_answer=True`, additionally generate after each
    patched forward and record whether the model emits the gold answer.
    The same `model.generate()` call provides log P(####) (from step-0
    logits) so no extra forward is needed. Per-layer behavioral success is
    saved alongside the log-P(####) recovery curve.
    """
    device = next(model.parameters()).device
    # cot_start patches at the LAST token of a NO-donor-CoT prompt (= end of
    # the assistant header, the slot where the model would begin writing its
    # own CoT). For every other patch_position the prompt includes the donor
    # CoT prefix so the readout at last token is "model about to emit ####".
    if patch_positions == "cot_start":
        source_answer_prefix = ""
        target_answer_prefix = ""
    else:
        # If the pair carries a target-specific CoT prefix (within-template
        # contrasts where each side ran with its own donor CoT), honor it.
        # Else fall back to source_cot_prefix for both — the legacy
        # shared-CoT design.
        target_cot_prefix = row.get("target_cot_prefix", None)
        if not isinstance(target_cot_prefix, str) or not target_cot_prefix:
            target_cot_prefix = row["source_cot_prefix"]
        source_answer_prefix = row["source_cot_prefix"]
        target_answer_prefix = target_cot_prefix
    source_prompt, source_q_start, source_q_end = build_prompt_and_question_span(
        row["source_question"], tokenizer, answer_prefix=source_answer_prefix
    )
    target_prompt, target_q_start, target_q_end = build_prompt_and_question_span(
        row["target_question"], tokenizer, answer_prefix=target_answer_prefix
    )
    source_inputs = encode_prompt(source_prompt, tokenizer, device)
    target_inputs = encode_prompt(target_prompt, tokenizer, device)

    tok_hash_ids = tokenizer.encode("####", add_special_tokens=False)
    if len(tok_hash_ids) != 1:
        raise ValueError(f"Expected single-token ####, got token ids {tok_hash_ids}")
    tok_hash = tok_hash_ids[0]

    # Reuse the unpatched #### logprobs from the baseline cache when present.
    # `compute_cot_swap_baselines` already saved both sides, so re-running
    # `hash_logprob` here is a pure waste of a forward pass per direction.
    # Fall back to inline computation only when no cache row exists.
    if (
        cached_baseline is not None
        and "source_hash_logprob" in cached_baseline
        and "target_hash_logprob" in cached_baseline
    ):
        source_hash_lp = float(cached_baseline["source_hash_logprob"])
        target_hash_lp = float(cached_baseline["target_hash_logprob"])
    else:
        source_hash_lp = hash_logprob(model, source_inputs, tok_hash)
        target_hash_lp = hash_logprob(model, target_inputs, tok_hash)
    denom = source_hash_lp - target_hash_lp  # always source-minus-target for traceability

    source_input_len = int(source_inputs["input_ids"].shape[1])
    target_input_len = int(target_inputs["input_ids"].shape[1])
    if direction == "restore":
        donor_inputs, recv_inputs = source_inputs, target_inputs
        donor_idx, recv_idx = patch_indices(
            source_q_start, source_q_end, target_q_start, target_q_end,
            patch_positions, allow_truncated_span=allow_truncated_span,
            source_input_len=source_input_len, target_input_len=target_input_len,
        )
        baseline_lp, other_lp   = target_hash_lp, source_hash_lp
    elif direction == "disrupt":
        donor_inputs, recv_inputs = target_inputs, source_inputs
        donor_idx, recv_idx = patch_indices(
            target_q_start, target_q_end, source_q_start, source_q_end,
            patch_positions, allow_truncated_span=allow_truncated_span,
            source_input_len=target_input_len, target_input_len=source_input_len,
        )
        baseline_lp, other_lp   = source_hash_lp, target_hash_lp
    else:
        raise ValueError(f"Unknown direction: {direction}")
    norm_denom = other_lp - baseline_lp

    num_layers = model.config.num_hidden_layers
    active_layers = layers if layers else list(range(num_layers))
    patched_hash_logprobs = np.full(num_layers, np.nan)
    readout_deltas = np.full(num_layers, np.nan)
    effects = np.full(num_layers, np.nan)
    # Behavioral metric (only populated when measure_final_answer=True).
    final_answer_success = np.full(num_layers, np.nan)
    hash_emitted = np.full(num_layers, np.nan)
    gold_immediate = np.full(num_layers, np.nan)
    emitted_answers = [None] * num_layers
    patched_gold_logprobs = np.full(num_layers, np.nan)
    normalized_gold_effects = np.full(num_layers, np.nan)
    unpatched_source_gold_lp = float("nan")
    unpatched_target_gold_lp = float("nan")
    source_minus_target_gold_lp = float("nan")
    # The recipient's gold answer = the source's gold (both share the same
    # underlying problem; only the inserted clause differs). For restore,
    # recipient=target; for disrupt, recipient=source. Either way, gold is
    # the same numerical answer.
    gold_answer = row.get("target_answer", row.get("source_answer"))

    skipped_mismatched_span = len(donor_idx) == 0
    # Gap filter:
    #   For all positions EXCEPT cot_start, filter on the hash-logprob gap
    #     (denom = source_hash_lp − target_hash_lp). This selects pairs where
    #     the model differentiates source from target on log P(####) at the
    #     readout — i.e., pairs that exhibit the cliff.
    #   For cot_start, the prompts have NO donor CoT, so log P(####) at the
    #     last token is ~−30 for both sides → denom near zero, the filter
    #     would drop almost every row. Use the GOLD-logprob gap from the
    #     baseline cache instead (which IS meaningful: source's gold lp >>
    #     target's gold lp on the cliff pairs). Requires baseline cache; if
    #     missing, fall back to no filter (effectively `min_source_target_gap=
    #     -inf`) and warn the caller in the result row.
    if patch_positions == "cot_start":
        # Prefer own-CoT gold gap (each side measured with its own CoT,
        # captures the natural cliff). Fall back to shared-donor-CoT gap for
        # backward compat with old baseline files that don't have the
        # own-CoT fields. Fall back further to "no filter" if no baseline.
        #
        # Filter on the ABSOLUTE value of the gap: the sign of
        # source_minus_target depends on which side is the natural
        # gold-emitter, and that inverts across contrasts. For
        # filler_vs_noop_clean the source (filler) naturally emits gold →
        # gap is large positive. For p1_vs_padded_symbolic the target (p1)
        # naturally emits its own gold while source (padded_symbolic)
        # doesn't → gap is large negative. Both produce a cliff under
        # patching; the filter should select pairs by magnitude of the
        # natural source/target differentiation, not its sign.
        cot_start_gap = None
        if cached_baseline is not None:
            v = cached_baseline.get("own_cot_source_minus_target_gold_logprob")
            if v is not None and not (isinstance(v, float) and v != v):  # not NaN
                cot_start_gap = float(v)
            elif "source_minus_target_gold_logprob" in cached_baseline:
                cot_start_gap = float(
                    cached_baseline["source_minus_target_gold_logprob"]
                )
        if cot_start_gap is not None:
            skipped_small_gap = abs(cot_start_gap) < min_source_target_gap
        else:
            skipped_small_gap = False
    else:
        skipped_small_gap = denom < min_source_target_gap

    if not skipped_small_gap and not skipped_mismatched_span:
        # When measuring final-answer recovery in logP, we need unpatched
        # gold-logprob baselines from source and target so we can normalize
        # per-layer patched gold-logprob the same way `effects` normalizes
        # patched hash-logprob. Prefer the precomputed cache (one row per
        # pair, shared across layer shards and directions); fall back to
        # inline computation only when no cache entry is present.
        if measure_final_answer and gold_answer is not None:
            if cached_baseline is not None:
                unpatched_source_gold_lp = float(
                    cached_baseline["unpatched_source_gold_logprob"]
                )
                unpatched_target_gold_lp = float(
                    cached_baseline["unpatched_target_gold_logprob"]
                )
                source_minus_target_gold_lp = (
                    unpatched_source_gold_lp - unpatched_target_gold_lp
                )
            else:
                from run_noop_activation_patching import (
                    _greedy_generate,
                    _gold_answer_logprob_from_generation_scores,
                )
                src_prompt_ids_list = source_inputs["input_ids"][0].detach().cpu().tolist()
                tgt_prompt_ids_list = target_inputs["input_ids"][0].detach().cpu().tolist()
                with torch.no_grad():
                    src_cont_ids, _, src_scores = _greedy_generate(
                        model, tokenizer, source_inputs,
                        max_new_tokens=max_new_tokens, return_scores=True,
                    )
                    tgt_cont_ids, _, tgt_scores = _greedy_generate(
                        model, tokenizer, target_inputs,
                        max_new_tokens=max_new_tokens, return_scores=True,
                    )
                unpatched_source_gold_lp = float(
                    _gold_answer_logprob_from_generation_scores(
                        model, tokenizer, src_prompt_ids_list, src_cont_ids,
                        str(gold_answer), src_scores,
                    )
                )
                unpatched_target_gold_lp = float(
                    _gold_answer_logprob_from_generation_scores(
                        model, tokenizer, tgt_prompt_ids_list, tgt_cont_ids,
                        str(gold_answer), tgt_scores,
                    )
                )
                source_minus_target_gold_lp = (
                    unpatched_source_gold_lp - unpatched_target_gold_lp
                )

        # Single donor forward saves the chosen activation at every active
        # layer, leaving each tensor on its native device.
        remove_all_hooks(model)
        saved = {}
        save_handles = []
        for layer_idx in active_layers:
            module, _ = layer_module(model, scope, layer_idx)
            save_handles.append(
                module.register_forward_hook(make_save_hook(saved, layer_idx))
            )
        with torch.no_grad():
            model(**donor_inputs)
        for h in save_handles:
            h.remove()

        for layer_idx in active_layers:
            module, _ = layer_module(model, scope, layer_idx)
            patch_hook = module.register_forward_hook(
                make_cross_position_patching_hook(
                    saved[layer_idx], donor_idx, recv_idx
                )
            )
            if measure_final_answer:
                gen = generate_and_score(
                    model, tokenizer, recv_inputs,
                    gold_answer=gold_answer, tok_hash=tok_hash,
                    max_new_tokens=max_new_tokens,
                )
                patched_hash_lp = gen["patched_hash_logprob"]
                final_answer_success[layer_idx] = float(
                    gen["patched_final_answer_success"]
                )
                hash_emitted[layer_idx] = float(gen["patched_hash_emitted"])
                gold_immediate[layer_idx] = float(gen["patched_gold_immediate"])
                emitted_answers[layer_idx] = gen["patched_emitted_answer"]
                patched_gold_logprobs[layer_idx] = gen["patched_gold_logprob"]
                if (
                    not np.isnan(source_minus_target_gold_lp)
                    and abs(source_minus_target_gold_lp) > 1e-6
                ):
                    normalized_gold_effects[layer_idx] = (
                        (gen["patched_gold_logprob"] - unpatched_target_gold_lp)
                        / source_minus_target_gold_lp
                    )
            else:
                patched_hash_lp = hash_logprob(model, recv_inputs, tok_hash)
            patch_hook.remove()
            patched_hash_logprobs[layer_idx] = patched_hash_lp
            readout_deltas[layer_idx] = patched_hash_lp - baseline_lp
            effects[layer_idx] = (
                (patched_hash_lp - baseline_lp) / norm_denom
                if abs(norm_denom) > 1e-6
                else np.nan
            )

        remove_all_hooks(model)

    return {
        "patch_positions": patch_positions,
        "num_patched_tokens": int(len(donor_idx)),
        "allow_truncated_span": bool(allow_truncated_span),
        "source_q_start": int(source_q_start),
        "source_q_end": int(source_q_end),
        "target_q_start": int(target_q_start),
        "target_q_end": int(target_q_end),
        "source_question_span_len": int(source_q_end - source_q_start + 1),
        "target_question_span_len": int(target_q_end - target_q_start + 1),
        "source_tok_len": int(source_inputs["input_ids"].shape[1]),
        "target_tok_len": int(target_inputs["input_ids"].shape[1]),
        "hash_token_id": int(tok_hash),
        "readout_name": "log P(next token = ####)",
        "source_hash_logprob": source_hash_lp,
        "target_hash_logprob": target_hash_lp,
        "denom_source_minus_target": denom,
        "direction": direction,
        "baseline_hash_logprob": baseline_lp,
        "norm_denom": norm_denom,
        "patched_hash_logprobs": patched_hash_logprobs,
        "patched_readout_values": patched_hash_logprobs,
        "source_baseline_readout": source_hash_lp,
        "target_baseline_readout": target_hash_lp,
        "baseline_readout": baseline_lp,
        "source_target_gap": denom,
        "readout_deltas": readout_deltas,
        "normalized_effects": effects,
        "effects": effects,
        "skipped_small_gap": bool(skipped_small_gap),
        "skipped_mismatched_span": bool(skipped_mismatched_span),
        # Behavioral fields — NaN/None when measure_final_answer was False.
        "gold_answer":            str(gold_answer) if gold_answer is not None else None,
        "final_answer_success":   final_answer_success,
        "hash_emitted":           hash_emitted,
        "gold_immediate":         gold_immediate,
        "patched_emitted_answers": emitted_answers,
        # Final-answer recovery on logP (continuous analog of the binary
        # final_answer_success). Normalized the same way as `effects`
        # but on log P(gold) at the generated #### boundary.
        "patched_gold_logprobs":          patched_gold_logprobs,
        "normalized_gold_effects":        normalized_gold_effects,
        "unpatched_source_gold_logprob":  unpatched_source_gold_lp,
        "unpatched_target_gold_logprob":  unpatched_target_gold_lp,
        "source_minus_target_gold_logprob": source_minus_target_gold_lp,
    }


def compute_cot_swap_baselines(
    model, tokenizer, model_name, contrast,
    cot_source=None, max_new_tokens=1024, overwrite=False,
    min_source_target_gap=0.0, shard=None,
    padded_dataset=None, p1_value_subset=None, length_filter="all",
):
    """Precompute per-pair unpatched gold-logprob baselines, write JSONL.

    For each pair, runs two short (boundary-stopped) greedy generations —
    one from `source_question + source_cot_prefix`, one from
    `target_question + (target_cot_prefix or source_cot_prefix)` — and
    extracts log P(gold first content token) at the post-`####` position
    via `_gold_answer_logprob_from_generation_scores`.

    Result is keyed by (original_id, instance); both `restore` and
    `disrupt` directions later consume the same baselines (the unpatched
    source/target generations don't depend on direction).

    Writing is append-mode + resumable: rows already in the cache file are
    skipped unless `overwrite=True`.
    """
    from run_noop_activation_patching import (
        _greedy_generate, _gold_answer_logprob_from_generation_scores,
    )

    # Dispatch loader by contrast type. p1_vs_padded_symbolic uses a separate
    # pair-builder that joins gsm_symbolic_to_p1 against the chosen padded
    # dataset; cot_swap contrasts use the cross-eval CSVs from
    # disentangled_evaluation.
    if contrast == "p1_vs_padded_symbolic":
        subsets = _normalize_subsets(p1_value_subset)
        df = _load_p1_contrast_data(
            contrast, padded_dataset, model_name, p1_value_subset=subsets,
            length_filter=length_filter,
        )
    else:
        df = load_cot_swap_as_patching_pairs(source=contrast, cot_source=cot_source)
    if df.empty:
        print(f"No pairs to baseline for contrast={contrast}, cot_source={cot_source}")
        return

    out_path = baselines_path_for_cot_swap(
        model_name, contrast, cot_source,
        padded_dataset=padded_dataset,
        p1_value_subset=p1_value_subset,
        length_filter=length_filter,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Data-parallel sharding: only handle pairs where (sorted-position % N) == k.
    # Each shard writes its own file; the loader globs them on the consumer side.
    if shard is not None:
        k, n_shards = shard
        if not (0 <= k < n_shards):
            raise ValueError(f"--baselines_shard k/N requires 0 <= k < N; got {k}/{n_shards}")
        df = df.reset_index(drop=True)
        df = df.iloc[k::n_shards].reset_index(drop=True)
        out_path = out_path.with_name(
            f"{out_path.stem}.shard_{k}_of_{n_shards}{out_path.suffix}"
        )
        print(f"Shard {k}/{n_shards}: {len(df)} pairs → {out_path.name}")

    done = set()
    if not overwrite and out_path.exists():
        existing = load_cot_swap_baselines(out_path)
        done = set(existing.keys())
        print(f"Resuming baselines — {len(done)} pairs cached at {out_path}")
    if overwrite and out_path.exists():
        out_path.unlink()

    device = next(model.parameters()).device
    cot_note = f" (CoT from `{cot_source}`)" if cot_source and cot_source != contrast else ""
    print(f"Computing baselines for {len(df)} pairs in `{contrast}`{cot_note} → {out_path}")
    model.eval()

    mode = "w" if overwrite else "a"
    with open(out_path, mode) as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="baselines"):
            key = (int(row["original_id"]), int(row["instance"]))
            if key in done:
                continue
            target_cot_prefix = row.get("target_cot_prefix", None)
            if not isinstance(target_cot_prefix, str) or not target_cot_prefix:
                target_cot_prefix = row["source_cot_prefix"]

            # SHARED-donor-CoT prompts (used for the readout-position log-P(####)
            # baselines that match the patching prompts).
            src_prompt, _, _ = build_prompt_and_question_span(
                row["source_question"], tokenizer,
                answer_prefix=row["source_cot_prefix"],
            )
            tgt_prompt, _, _ = build_prompt_and_question_span(
                row["target_question"], tokenizer,
                answer_prefix=target_cot_prefix,
            )
            src_in = encode_prompt(src_prompt, tokenizer, device)
            tgt_in = encode_prompt(tgt_prompt, tokenizer, device)
            src_prompt_ids = src_in["input_ids"][0].detach().cpu().tolist()
            tgt_prompt_ids = tgt_in["input_ids"][0].detach().cpu().tolist()

            gold = str(row["target_answer"]) if pd.notna(row["target_answer"]) else None
            if gold is None:
                continue

            # Cache the unpatched #### logprobs (cheap single forward each) so
            # later patching runs don't redo them for the gap filter.
            tok_hash_ids = tokenizer.encode("####", add_special_tokens=False)
            tok_hash = tok_hash_ids[0]
            src_hash_lp = hash_logprob(model, src_in, tok_hash)
            tgt_hash_lp = hash_logprob(model, tgt_in, tok_hash)
            denom_hash = src_hash_lp - tgt_hash_lp

            # Shared-donor-CoT gold logprobs (each side's question + donor CoT).
            with torch.no_grad():
                src_cont_ids, _, src_scores = _greedy_generate(
                    model, tokenizer, src_in,
                    max_new_tokens=max_new_tokens, return_scores=True,
                )
                tgt_cont_ids, _, tgt_scores = _greedy_generate(
                    model, tokenizer, tgt_in,
                    max_new_tokens=max_new_tokens, return_scores=True,
                )
            src_gold_lp = float(_gold_answer_logprob_from_generation_scores(
                model, tokenizer, src_prompt_ids, src_cont_ids,
                gold, src_scores,
            ))
            tgt_gold_lp = float(_gold_answer_logprob_from_generation_scores(
                model, tokenizer, tgt_prompt_ids, tgt_cont_ids,
                gold, tgt_scores,
            ))

            # OWN-CoT gold logprobs: each question paired with the model's OWN
            # natural CoT on that question. These reflect the natural cliff —
            # source achieves gold (correct side), target fails (wrong side)
            # for cot_swap; padded gets its gold, p1 gets its gold for p1
            # contrast. Read by cot_start gap filter via
            # `own_cot_source_minus_target_gold_logprob`. NaN if loader
            # didn't supply per-side own CoTs.
            own_src_gold_lp = float("nan")
            own_tgt_gold_lp = float("nan")
            if "source_own_cot_prefix" in row and "target_own_cot_prefix" in row \
               and isinstance(row["source_own_cot_prefix"], str) \
               and isinstance(row["target_own_cot_prefix"], str) \
               and row["source_own_cot_prefix"] and row["target_own_cot_prefix"]:
                own_src_prompt, _, _ = build_prompt_and_question_span(
                    row["source_question"], tokenizer,
                    answer_prefix=row["source_own_cot_prefix"],
                )
                own_tgt_prompt, _, _ = build_prompt_and_question_span(
                    row["target_question"], tokenizer,
                    answer_prefix=row["target_own_cot_prefix"],
                )
                own_src_in = encode_prompt(own_src_prompt, tokenizer, device)
                own_tgt_in = encode_prompt(own_tgt_prompt, tokenizer, device)
                own_src_prompt_ids = own_src_in["input_ids"][0].detach().cpu().tolist()
                own_tgt_prompt_ids = own_tgt_in["input_ids"][0].detach().cpu().tolist()
                with torch.no_grad():
                    own_src_cont_ids, _, own_src_scores = _greedy_generate(
                        model, tokenizer, own_src_in,
                        max_new_tokens=max_new_tokens, return_scores=True,
                    )
                    own_tgt_cont_ids, _, own_tgt_scores = _greedy_generate(
                        model, tokenizer, own_tgt_in,
                        max_new_tokens=max_new_tokens, return_scores=True,
                    )
                own_src_gold_lp = float(_gold_answer_logprob_from_generation_scores(
                    model, tokenizer, own_src_prompt_ids, own_src_cont_ids,
                    gold, own_src_scores,
                ))
                own_tgt_gold_lp = float(_gold_answer_logprob_from_generation_scores(
                    model, tokenizer, own_tgt_prompt_ids, own_tgt_cont_ids,
                    gold, own_tgt_scores,
                ))

            rec = {
                "original_id":                     int(row["original_id"]),
                "instance":                        int(row["instance"]),
                "contrast":                        contrast,
                "cot_source":                      cot_source or contrast,
                "gold_answer":                     gold,
                "source_hash_logprob":             float(src_hash_lp),
                "target_hash_logprob":             float(tgt_hash_lp),
                "denom_source_minus_target_hash":  float(denom_hash),
                # Shared-donor-CoT gold baselines (existing semantics).
                "unpatched_source_gold_logprob":   float(src_gold_lp),
                "unpatched_target_gold_logprob":   float(tgt_gold_lp),
                "source_minus_target_gold_logprob": float(src_gold_lp - tgt_gold_lp),
                # OWN-CoT gold baselines (new). Used by the cot_start gap
                # filter. NaN when the loader didn't populate own-CoT columns.
                "own_cot_source_gold_logprob":     float(own_src_gold_lp),
                "own_cot_target_gold_logprob":     float(own_tgt_gold_lp),
                "own_cot_source_minus_target_gold_logprob": float(
                    own_src_gold_lp - own_tgt_gold_lp
                ),
            }
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
    print(f"Baselines → {out_path}")


def run_activation_patching(
    model,
    tokenizer,
    model_name,
    contrast,
    padded_dataset,
    p1_value_subset,
    scope,
    direction="restore",
    patch_positions="question_end",
    allow_truncated_span=False,
    layers=None,
    max_examples=None,
    min_source_target_gap=0.5,
    overwrite=False,
    length_filter="all",
    cot_source=None,
    measure_final_answer=False,
    max_new_tokens=1024,
    allow_no_baseline_cache=False,
    data_shard=None,
):
    if contrast == "p1_vs_padded_symbolic":
        subsets = _normalize_subsets(p1_value_subset)
        df = _load_p1_contrast_data(
            contrast, padded_dataset, model_name, p1_value_subset=subsets,
            length_filter=length_filter,
        )
    elif contrast in COT_SWAP_PATCHING_CONTRASTS:
        subsets = ["all"]
        df = load_cot_swap_as_patching_pairs(source=contrast, cot_source=cot_source)
        if cot_source is not None and cot_source != contrast:
            print(f"  hybrid: questions from `{contrast}`, donor CoT from `{cot_source}`")
    else:
        raise ValueError(
            f"Unknown contrast: {contrast}. Choices: p1_vs_padded_symbolic, "
            f"{sorted(COT_SWAP_PATCHING_CONTRASTS)}"
        )
    if max_examples is not None:
        df = df.head(max_examples).copy()

    # Data-parallel sharding: this task only processes pairs where
    # (position_in_sorted_df % N) == k. The write path gets a
    # `.shard_{k}_of_{N}` suffix so K parallel writers don't clobber each
    # other; the consumer-side loader globs both the unsharded path and
    # all sibling shards. Apply BEFORE the max_examples cap so the same
    # pair is never picked up by two shards.
    if data_shard is not None:
        k, n_shards = data_shard
        if not (0 <= k < n_shards):
            raise ValueError(
                f"--data_shard k/N requires 0 <= k < N; got {k}/{n_shards}"
            )
        df = df.reset_index(drop=True)
        df = df.iloc[k::n_shards].reset_index(drop=True)

    out_path = output_path(
        model_name, contrast, padded_dataset, subsets, scope,
        direction=direction, patch_positions=patch_positions,
        allow_truncated_span=allow_truncated_span, layers=layers,
        length_filter=length_filter, cot_source=cot_source,
    )
    out_path = _add_shard_suffix(out_path, data_shard)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink()

    # Load precomputed gold-logprob baselines for `--measure_final_answer`.
    # Keyed by (original_id, instance), shared across directions + layer
    # shards. Missing entries trigger inline computation (or a hard error
    # if --allow_no_baseline_cache wasn't passed).
    baseline_cache = {}
    if measure_final_answer and (
        contrast in COT_SWAP_PATCHING_CONTRASTS
        or contrast == "p1_vs_padded_symbolic"
    ):
        baselines_path = baselines_path_for_cot_swap(
            model_name, contrast, cot_source,
            padded_dataset=padded_dataset,
            p1_value_subset=p1_value_subset,
            length_filter=length_filter,
        )
        baseline_cache = load_cot_swap_baselines(baselines_path)
        if baseline_cache:
            print(
                f"Loaded {len(baseline_cache)} cached gold-logprob baselines "
                f"from {baselines_path}"
            )
        elif allow_no_baseline_cache:
            print(
                f"No baseline cache at {baselines_path}; gold-logprob "
                f"baselines will be computed inline (--allow_no_baseline_cache "
                f"set, expect 2 extra greedy generations per pair)."
            )
        else:
            raise SystemExit(
                "FATAL: --measure_final_answer requires a precomputed "
                "gold-logprob baseline cache, but none was found.\n"
                f"  Expected at: {baselines_path}\n\n"
                "Fix one of these:\n"
                "  (1) Run baselines first:\n"
                "        python run_cot_swap_activation_patching.py \\\n"
                f"          --contrast {contrast} "
                f"{'--cot_source ' + cot_source if cot_source and cot_source != contrast else ''} \\\n"
                "          --baselines_only [--max_new_tokens N]\n"
                "      (one task, baselines are shared across directions and "
                "layer shards).\n"
                "  (2) Or accept inline recomputation per layer shard:\n"
                "          --allow_no_baseline_cache"
            )

    # Resume key: legacy pairs use (original_id, instance); within-template
    # contrasts cartesian-product multiple targets per source, so they also
    # need target_instance in the key for uniqueness. We auto-detect by
    # checking whether the dataframe has the column.
    has_target_instance = "target_instance" in df.columns
    def _resume_key_from_record(r):
        if has_target_instance and r.get("target_instance") is not None:
            return (int(r["original_id"]), int(r["instance"]), int(r["target_instance"]))
        return (int(r["original_id"]), int(r["instance"]))

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                done.add(_resume_key_from_record(r))
        print(f"Resuming - {len(done)} examples already done")

    print(
        f"Loaded {len(df)} examples ({df['original_id'].nunique()} templates), "
        f"scope={scope}, direction={direction}, padded_dataset={padded_dataset}, "
        f"length_filter={length_filter}, patch_positions={patch_positions}, "
        f"p1_value_subset={'+'.join(subsets)}, "
        f"min_source_target_gap={min_source_target_gap}"
    )
    model.eval()
    skipped_small_gap = 0
    skipped_mismatched_span = 0
    with open(out_path, "a") as f_out:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=scope):
            if has_target_instance and not pd.isna(row.get("target_instance")):
                key = (int(row["original_id"]), int(row["instance"]), int(row["target_instance"]))
            else:
                key = (int(row["original_id"]), int(row["instance"]))
            if key in done:
                continue
            cached = baseline_cache.get(
                (int(row["original_id"]), int(row["instance"]))
            ) if baseline_cache else None
            metrics = patch_one(
                model,
                tokenizer,
                row,
                scope=scope,
                layers=layers,
                min_source_target_gap=min_source_target_gap,
                direction=direction,
                patch_positions=patch_positions,
                allow_truncated_span=allow_truncated_span,
                measure_final_answer=measure_final_answer,
                max_new_tokens=max_new_tokens,
                cached_baseline=cached,
            )
            if metrics.pop("skipped_small_gap"):
                skipped_small_gap += 1
                continue
            if metrics.pop("skipped_mismatched_span"):
                skipped_mismatched_span += 1
                continue
            result = {
                "original_id": int(row["original_id"]),
                "instance": int(row["instance"]),
                "contrast": contrast,
                "padded_dataset": padded_dataset,
                "length_filter": length_filter,
                "p1_value_subset": subsets,
                "scope": scope,
                "source_label": row["source_label"],
                "target_label": row["target_label"],
                "source_answer": str(row["source_answer"]),
                "target_answer": str(row["target_answer"]),
                "len_delta_after": int(row["len_delta_after"]),
                "source_question": row["source_question"],
                "target_question": row["target_question"],
                "source_cot_prefix": row["source_cot_prefix"],
                **{k: to_serializable(v) for k, v in metrics.items()},
            }
            # Within-template contrasts carry per-side CoT + target_instance.
            # Surface them in the JSONL so downstream analyses can recover the
            # exact pair the patching ran against.
            if "target_cot_prefix" in row and isinstance(row["target_cot_prefix"], str) and row["target_cot_prefix"]:
                result["target_cot_prefix"] = row["target_cot_prefix"]
            if "target_instance" in row and pd.notna(row["target_instance"]):
                result["target_instance"] = int(row["target_instance"])
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            # Drain the CUDA allocator's cache between pairs. patch_one
            # accumulates per-layer activations and per-pair `saved` dicts;
            # without an explicit empty_cache() the allocator can hold onto
            # large pools, fragment HBM, and (under cluster reapers that watch
            # for "hung GPU" signals) trigger an external SIGTERM on long
            # runs. The cost is a few hundred ms per pair; the benefit is a
            # stable memory footprint that doesn't drift across hundreds of
            # iterations.
            torch.cuda.empty_cache()

    if skipped_small_gap:
        # The gap underlying the filter is patch_positions-dependent: for
        # cot_start there's no donor CoT and #### isn't meaningful at the
        # readout, so the filter uses |gold-logprob gap| from the baseline
        # cache instead of the #### readiness gap used elsewhere.
        gap_kind = (
            "|gold-logprob gap|"
            if patch_positions == "cot_start"
            else "source-target #### readiness gap"
        )
        print(
            f"Skipped {skipped_small_gap} examples with "
            f"{gap_kind} < {min_source_target_gap}"
        )
    if skipped_mismatched_span:
        print(
            f"Skipped {skipped_mismatched_span} examples with mismatched "
            f"question-span token lengths. Use --allow_truncated_span to patch "
            f"the shared prefix instead."
        )
    print(f"Results -> {out_path}")
    return out_path


def _short_label(label):
    return str(label).replace("padded_", "")


def _gap_filter_tag(min_source_target_gap, default=0.5):
    """Filename suffix for stricter-than-default denom filters."""
    if min_source_target_gap is None or min_source_target_gap <= default:
        return ""
    return f"_gap_gt_{str(min_source_target_gap).replace('.', 'p')}"


def _ratio_of_means_bootstrap(delta_patched_t, delta_full_t,
                              n_boot=1000, ci=0.95, seed=0):
    """Bootstrap mean(patched-baseline) / mean(other-baseline) by resampling templates.

    Per-row ratios are heavy-tailed when ``denom`` is small; resampling templates
    and forming the ratio of *means* keeps the statistic bounded and gives the
    intuitive interpretation (1 = full recovery, 0 = no recovery).
    """
    rng = np.random.default_rng(seed)
    n = delta_patched_t.shape[0]
    point = delta_patched_t.mean(axis=0) / delta_full_t.mean()
    if n == 1:
        return point, point.copy(), point.copy()
    boot = np.empty((n_boot, delta_patched_t.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = delta_patched_t[idx].mean(axis=0) / delta_full_t[idx].mean()
    alpha = (1 - ci) / 2
    lo = np.quantile(boot, alpha, axis=0)
    hi = np.quantile(boot, 1 - alpha, axis=0)
    return point, lo, hi


def _shard_aware_files(path):
    """Return all files matching `path` and its `.shard_*_of_*` siblings.

    Lets data-parallel writers split rows across `restore.shard_0_of_4.jsonl`,
    `restore.shard_1_of_4.jsonl`, … and have the loader stitch them back
    without an explicit merge step.
    """
    files = []
    if path.exists():
        files.append(path)
    files.extend(sorted(path.parent.glob(
        f"{path.stem}.shard_*_of_*{path.suffix}"
    )))
    return files


def _load_effects(path, min_source_target_gap):
    """Return dict with layers, mean, ci, counts, and source/target label names."""
    files = _shard_aware_files(path)
    if not files:
        return None
    rows = []
    for fp in files:
        rows.extend(json.loads(line) for line in open(fp) if line.strip())
    rows = [
        r for r in rows
        if float(r.get("denom_source_minus_target", 0.0)) >= min_source_target_gap
    ]
    if not rows:
        return None
    patched = np.array([r["patched_hash_logprobs"] for r in rows], dtype=float)
    baseline = np.array([r["baseline_hash_logprob"] for r in rows], dtype=float)
    norm_denom = np.array([r["norm_denom"] for r in rows], dtype=float)
    cluster_ids = np.array([r["original_id"] for r in rows])
    finite_layers = np.isfinite(patched).any(axis=0)
    if not finite_layers.any():
        return None
    patched = patched[:, finite_layers]
    block_idx = np.arange(finite_layers.shape[0])[finite_layers]
    layers_arr = block_idx + 1
    delta_patched = patched - baseline[:, None]
    template_ids, delta_patched_t = _template_means(delta_patched, cluster_ids)
    _, norm_denom_t = _template_means(norm_denom, cluster_ids)
    mean, lo, hi = _ratio_of_means_bootstrap(
        delta_patched_t, norm_denom_t, n_boot=1000, ci=0.95
    )
    return {
        "layers": layers_arr,
        "mean": mean,
        "lo": lo,
        "hi": hi,
        "n_rows": len(rows),
        "n_templates": len(template_ids),
        "num_layers": int(finite_layers.shape[0]),
        "source_name": _short_label(rows[0].get("source_label", "source")),
        "target_name": _short_label(rows[0].get("target_label", "target")),
    }


def plot_restore_disrupt_overlay(
    model_name,
    contrast,
    padded_dataset,
    p1_value_subset,
    scope,
    patch_positions="question_span",
    allow_truncated_span=False,
    layers=None,
    min_source_target_gap=0.5,
    length_filter="all",
    cot_source=None,
):
    """Overlay restore + disrupt normalized patching effects on shared axes."""
    subsets = _normalize_subsets(p1_value_subset)
    restore_path = output_path(
        model_name, contrast, padded_dataset, subsets, scope,
        direction="restore", patch_positions=patch_positions,
        allow_truncated_span=allow_truncated_span, layers=layers,
        length_filter=length_filter, cot_source=cot_source,
    )
    disrupt_path = output_path(
        model_name, contrast, padded_dataset, subsets, scope,
        direction="disrupt", patch_positions=patch_positions,
        allow_truncated_span=allow_truncated_span, layers=layers,
        length_filter=length_filter, cot_source=cot_source,
    )
    restore = _load_effects(restore_path, min_source_target_gap)
    disrupt = _load_effects(disrupt_path, min_source_target_gap)
    if restore is None or disrupt is None:
        print(
            f"Overlay skipped (missing data): restore={restore is not None}, "
            f"disrupt={disrupt is not None} at {restore_path.parent}"
        )
        return

    num_layers = restore["num_layers"]
    # In restore: donor=source, receiver=target. In disrupt: donor=target, receiver=source.
    r_donor, r_recv = restore["source_name"], restore["target_name"]
    n_donor, n_recv = disrupt["target_name"], disrupt["source_name"]

    fig, ax = plt.subplots(figsize=(12, 3.8))
    restore_color = "#1b7837"  # green: recovery
    disrupt_color = "#762a83"    # purple: disruption
    ax.plot(restore["layers"], restore["mean"], color=restore_color, lw=1.8,
            label=f"restore ({r_donor}→{r_recv})")
    ax.fill_between(restore["layers"], restore["lo"], restore["hi"],
                    color=restore_color, alpha=0.18)
    ax.plot(disrupt["layers"], disrupt["mean"], color=disrupt_color, lw=1.8,
            label=f"disrupt ({n_donor}→{n_recv})")
    ax.fill_between(disrupt["layers"], disrupt["lo"], disrupt["hi"],
                    color=disrupt_color, alpha=0.18)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axhline(1, color="black", lw=0.8, alpha=0.3)
    ax.set_ylabel("Normalized patching effect")
    _style_layer_axis(ax, num_layers, stage_boundaries=PATCHING_STAGE_BOUNDARIES)
    ax.legend(fontsize=9.5, loc="center left", frameon=True)
    plt.tight_layout()
    save_path = restore_path.with_name("overlay.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Overlay -> {save_path}")


def plot_activation_patching(
    model_name,
    contrast,
    padded_dataset,
    p1_value_subset,
    scope,
    direction="restore",
    patch_positions="question_end",
    allow_truncated_span=False,
    layers=None,
    min_source_target_gap=0.5,
    length_filter="all",
    cot_source=None,
):
    subsets = _normalize_subsets(p1_value_subset)
    path = output_path(
        model_name, contrast, padded_dataset, subsets, scope,
        direction=direction, patch_positions=patch_positions,
        allow_truncated_span=allow_truncated_span, layers=layers,
        length_filter=length_filter, cot_source=cot_source,
    )
    files = _shard_aware_files(path)
    if not files:
        print(f"Results not found: {path}")
        return
    rows = []
    for fp in files:
        rows.extend(json.loads(line) for line in open(fp) if line.strip())
    if len(files) > 1:
        print(f"Loaded {len(rows)} rows across {len(files)} shard files")
    before = len(rows)
    rows = [
        r for r in rows
        if float(r.get("denom_source_minus_target", 0.0)) >= min_source_target_gap
    ]
    if not rows:
        print(
            f"No rows remain after filtering source-target gap >= "
            f"{min_source_target_gap} (before={before})"
        )
        return
    if len(rows) != before:
        print(
            f"Plotting {len(rows)} / {before} rows after filtering "
            f"source-target gap >= {min_source_target_gap}"
        )
    patched = np.array([r["patched_hash_logprobs"] for r in rows], dtype=float)
    source = np.array([r["source_hash_logprob"] for r in rows], dtype=float)
    target = np.array([r["target_hash_logprob"] for r in rows], dtype=float)
    baseline = np.array([r["baseline_hash_logprob"] for r in rows], dtype=float)
    norm_denom = np.array([r["norm_denom"] for r in rows], dtype=float)
    cluster_ids = np.array([r["original_id"] for r in rows])

    # Dataset names for the legend / effect formula. Strip the "padded_" prefix
    # since padding is a length-matching detail, not a different dataset.
    def _short(label):
        return str(label).replace("padded_", "")
    source_name = _short(rows[0].get("source_label", "source"))
    target_name = _short(rows[0].get("target_label", "target"))
    finite_layers = np.isfinite(patched).any(axis=0)
    if not finite_layers.any():
        print(f"No finite layer results to plot in {path}")
        return
    patched = patched[:, finite_layers]
    # Block-output index i -> template_similarity L(i+1): residual after i+1
    # blocks have processed. Aligns stage markers across the four experiments.
    block_idx = np.arange(finite_layers.shape[0])[finite_layers]
    layers_arr = block_idx + 1
    num_layers = int(finite_layers.shape[0])

    # Template-level aggregation matches the lens panel: rows within the same
    # template are not independent, so average within template first. The
    # normalized effect uses ratio-of-means (mean per-template delta divided by
    # mean per-template norm_denom) rather than mean-of-ratios, which blows up
    # when individual rows have small denominators — see the cot_end within-
    # template plots where a handful of small-denom outliers dragged the
    # template mean to ~3 even though the row-level median sat near 1.
    delta_patched = patched - baseline[:, None]
    template_ids, delta_patched_t = _template_means(delta_patched, cluster_ids)
    _, norm_denom_t = _template_means(norm_denom, cluster_ids)
    _, patched_template = _template_means(patched, cluster_ids)
    n_clusters = len(template_ids)

    effect_mean, effect_lo, effect_hi = _ratio_of_means_bootstrap(
        delta_patched_t, norm_denom_t, n_boot=1000, ci=0.95
    )
    patched_mean = patched_template.mean(axis=0)
    patched_lo, patched_hi = _bootstrap_ci(patched_template, n_boot=1000, ci=0.95)
    source_mean = float(np.nanmean(source))
    target_mean = float(np.nanmean(target))

    if direction == "restore":
        patched_label = f"patched {target_name} ({source_name}→{target_name})"
        effect_ylabel = (
            f"(patched − {target_name}) / ({source_name} − {target_name})"
        )
    else:
        patched_label = f"patched {source_name} ({target_name}→{source_name})"
        effect_ylabel = (
            f"(patched − {source_name}) / ({target_name} − {source_name})"
        )

    fig0, ax0 = plt.subplots(figsize=(12, 3.6))
    ax0.plot(layers_arr, patched_mean, color="#333333", lw=1.8, label=patched_label)
    ax0.fill_between(layers_arr, patched_lo, patched_hi, color="#333333", alpha=0.18)
    ax0.axhline(source_mean, color="#2166ac", lw=0.9, ls=":",
                label=f"{source_name} baseline")
    ax0.axhline(target_mean, color="#d6604d", lw=0.9, ls=":",
                label=f"{target_name} baseline")
    ax0.set_ylabel("Mean log P(answer marker)")
    _style_layer_axis(ax0, num_layers,
                      stage_boundaries=PATCHING_STAGE_BOUNDARIES)
    ax0.legend(fontsize=9.5, loc="center left", frameon=True)
    plt.tight_layout()
    gap_tag = _gap_filter_tag(min_source_target_gap)
    raw_path = path.with_name(path.stem + gap_tag + "_logprob.png")
    plt.savefig(raw_path, dpi=200, bbox_inches="tight")
    plt.close(fig0)

    fig1, ax1 = plt.subplots(figsize=(12, 3.6))
    ax1.plot(layers_arr, effect_mean, color="#333333", lw=1.8,
             label="normalized patching effect")
    ax1.fill_between(layers_arr, effect_lo, effect_hi, color="#333333", alpha=0.18)
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.axhline(1, color="black", lw=0.8, alpha=0.3)
    ax1.set_ylabel(effect_ylabel)
    _style_layer_axis(ax1, num_layers,
                      stage_boundaries=PATCHING_STAGE_BOUNDARIES)
    ax1.legend(fontsize=9.5, loc="center left", frameon=True)
    plt.tight_layout()
    save_path = path.with_name(path.stem + gap_tag + ".png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig1)
    print(
        f"Plots -> {save_path}, {raw_path} "
        f"(N={len(rows)} rows, {n_clusters} templates)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        choices=list(MODEL_NAME_MAP.keys()),
    )
    parser.add_argument(
        "--contrast",
        choices=[
            "p1_vs_padded_symbolic",
            "filler_vs_noop_clean",
            "filler_df_clean_vs_noop_clean_fail",
            "filler_df_correct_vs_noop_clean_wrong",
            "filler_clean_vs_noop_clean_fail",
            "filler_correct_vs_noop_clean_wrong",
            "noop_clean_within_template",
            "noop_clean_within_template_clean_vs_fail",
        ],
        default="p1_vs_padded_symbolic",
        help=(
            "p1_vs_padded_symbolic: patch P1 question activations into the "
            "padded-symbolic prompt under the padded-symbolic CoT. "
            "filler_vs_noop_clean: patch filler-question activations into the "
            "noop_clean prompt under the filler CoT (loads "
            "results/disentangled_evaluation/cot_swap_filler_vs_noop_clean/). "
            "filler_df_clean_vs_noop_clean_fail: patch filler_df question "
            "activations (rows where the model accepted the sym donor CoT "
            "cleanly, correct ∧ immediate ####) into the noop_clean prompt "
            "(rows where the donor CoT failed to recover and the model kept "
            "reasoning, ~correct ∧ ~immediate ####). Both sides share the "
            "donor sym CoT. Loads results/disentangled_evaluation/"
            "cot_swap_filler_df_clean_vs_noop_clean_fail/. "
            "filler_df_correct_vs_noop_clean_wrong: looser variant of the "
            "above — drops the ####-immediate requirement on both sides, "
            "selecting on swap_correct alone. Loads "
            "results/disentangled_evaluation/"
            "cot_swap_filler_df_correct_vs_noop_clean_wrong/."
        ),
    )
    parser.add_argument(
        "--padded_dataset",
        choices=[
            "gsm_padded_symbolic",
            "gsm_padded_symbolic_p1_aligned",
            "legacy_symbolic_padded",
        ],
        default="gsm_padded_symbolic_p1_aligned",
        help=(
            "Padded sym dataset for p1_vs_padded_symbolic. Use "
            "gsm_padded_symbolic_p1_aligned (entity-aligned to p1) — the "
            "loader rejects gsm_padded_symbolic for this contrast. "
            "legacy_symbolic_padded loads pre-filtered legacy CSVs; "
            "--length_filter selects delta0 or lt2."
        ),
    )
    parser.add_argument(
        "--length_filter",
        choices=["all", "delta0", "lt2"],
        default="all",
        help=(
            "Length-delta filter on (oid, inst) pairs for "
            "p1_vs_padded_symbolic: delta0 → tok_len(source)==tok_len(p1); "
            "lt2 → |Δ|<2; all → no filter (default). For "
            "legacy_symbolic_padded, selects which pre-filtered legacy file "
            "to load (delta0 or lt2; `all` is not available)."
        ),
    )
    parser.add_argument(
        "--p1_value_subset",
        nargs="+",
        choices=["all", "no_value", "with_value", "value_present", "value_absent"],
        default=["no_value"],
    )
    parser.add_argument("--scope", choices=["layer", "mlp", "attn_output"], default="layer")
    parser.add_argument(
        "--direction", choices=["restore", "disrupt"], default="restore",
        help=(
            "restore: patch source question activations INTO target, measure target's "
            "P(####) recovery. disrupt: patch target question activations INTO source, "
            "measure source's P(####) degradation. Use 'disrupt' to avoid floor effects "
            "when target P(####) is near 0."
        ),
    )
    parser.add_argument(
        "--patch_positions",
        choices=["question_end", "question_span", "cot_end", "cot_start"],
        default="question_end",
        help=(
            "question_end patches only the final question token. question_span patches "
            "the full aligned question token span, skipping examples whose source and "
            "target question spans have different token counts unless "
            "--allow_truncated_span is set. cot_end patches only the final token of "
            "the shared CoT prefix (the position whose next-token logits are read out). "
            "cot_start patches the final token of the prompt WITH NO donor CoT — i.e., "
            "the end of the assistant header, where the model would begin generating "
            "its own CoT; usually paired with --measure_final_answer because log P(####) "
            "at that slot is uninformative."
        ),
    )
    parser.add_argument(
        "--allow_truncated_span",
        action="store_true",
        help=(
            "For --patch_positions question_span, patch the shared prefix when source "
            "and target question spans have different token counts."
        ),
    )
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument(
        "--min_source_target_gap",
        type=float,
        default=0.5,
        help=(
            "Pair-selection filter. Semantics depend on --patch_positions:\n"
            "  question_end / question_span / cot_end:  keep pairs where "
            "source log P(####) − target log P(####) >= this value, measured "
            "with the donor CoT in the prompt. Avoids unstable normalized "
            "effects on pairs that are already ####-ready.\n"
            "  cot_start (no donor CoT in prompt):  log P(####) at the last "
            "token is uniformly ~−30 nats, so the gap is uninformative. The "
            "filter switches to GOLD logprob gap, read from the "
            "`source_minus_target_gold_logprob` field of the baseline cache. "
            "Pass a value > 0 (e.g. 5) to keep only pairs where the model "
            "clearly resolves source's gold but not target's; pass a large "
            "negative number to disable the filter entirely. Requires "
            "baselines computed with --baselines_only."
        ),
    )
    parser.add_argument(
        "--cot_source",
        choices=sorted(COT_SWAP_PATCHING_CONTRASTS) + [None],
        default=None,
        help=(
            "Override source of the donor CoT (`source_cot_prefix`). Defaults "
            "to --contrast. Pass e.g. `noop_clean` to use symbolic's clean "
            "CoT while keeping --contrast's questions for length control. "
            "Output gets a `_cot-<cot_source>` directory suffix."
        ),
    )
    parser.add_argument(
        "--measure_final_answer",
        action="store_true",
        help=(
            "Also generate after each patched forward and record whether "
            "the model emits the correct final answer. Adds a behavioral "
            "(binary success-rate) curve alongside the log P(####) recovery. "
            "Slower per-layer because each layer's intervention is now a "
            "full short generation instead of a single forward pass."
        ),
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Generation budget when --measure_final_answer. Generation "
             "early-stops at the first numeric token after `####` via "
             "AnswerBoundaryStoppingCriteria, so the budget is only the hard "
             "ceiling for runaways that never emit `####`.",
    )
    parser.add_argument(
        "--baselines_only",
        action="store_true",
        help=(
            "Precompute the per-pair unpatched gold-logprob baselines and "
            "exit. One pass over all pairs (no layer loop, no patching). "
            "Cache lives at "
            "results/cot_swap_activation_patching/<model>/<contrast[_cot-...]>/baselines.jsonl "
            "and is shared by both directions and any future layer shards."
        ),
    )
    parser.add_argument(
        "--baselines_shard",
        type=str,
        default=None,
        help=(
            "Data-parallel shard spec 'k/N' (0-indexed). When set with "
            "--baselines_only, only handles pairs where index%%N == k and "
            "writes to baselines.shard_{k}_of_{N}.jsonl. The loader globs "
            "all shards, so consumers don't need to merge."
        ),
    )
    parser.add_argument(
        "--data_shard",
        type=str,
        default=None,
        help=(
            "Data-parallel shard spec 'k/N' (0-indexed) for the patching "
            "loop. Each shard handles pairs where index%%N == k and writes "
            "to {direction}.shard_{k}_of_{N}.jsonl. Plotters glob both the "
            "unsharded file and sibling shards, so no merge is needed."
        ),
    )
    parser.add_argument(
        "--allow_no_baseline_cache",
        action="store_true",
        help=(
            "When --measure_final_answer is on, allow the per-layer patching "
            "to compute baselines inline if the cache is missing. Slower "
            "(2 extra generations per pair per shard); prefer precomputing "
            "with --baselines_only instead."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument(
        "--no_plot_after_run",
        action="store_true",
        help=(
            "Skip post-run plotting. Use for per-layer shards (each shard's "
            "arrays have only one finite layer, so a per-shard plot is "
            "degenerate). Merge shards first, then run with --plot_only."
        ),
    )
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]
    subsets = _normalize_subsets(args.p1_value_subset)

    if args.baselines_only:
        if args.contrast not in COT_SWAP_PATCHING_CONTRASTS and args.contrast != "p1_vs_padded_symbolic":
            raise SystemExit(
                f"--baselines_only only supported for cot_swap and "
                f"p1_vs_padded_symbolic contrasts; got {args.contrast}"
            )
        shard = None
        if args.baselines_shard:
            k_str, n_str = args.baselines_shard.split("/")
            shard = (int(k_str), int(n_str))
        model, tokenizer = load_model(args.model_id)
        compute_cot_swap_baselines(
            model, tokenizer, model_name,
            contrast=args.contrast,
            cot_source=args.cot_source,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
            shard=shard,
            padded_dataset=args.padded_dataset,
            p1_value_subset=subsets,
            length_filter=args.length_filter,
        )
        sys.exit(0)

    data_shard = None
    if args.data_shard:
        k_str, n_str = args.data_shard.split("/")
        data_shard = (int(k_str), int(n_str))

    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        run_activation_patching(
            model,
            tokenizer,
            model_name,
            contrast=args.contrast,
            padded_dataset=args.padded_dataset,
            p1_value_subset=subsets,
            scope=args.scope,
            direction=args.direction,
            patch_positions=args.patch_positions,
            allow_truncated_span=args.allow_truncated_span,
            layers=args.layers,
            max_examples=args.max_examples,
            min_source_target_gap=args.min_source_target_gap,
            overwrite=args.overwrite,
            length_filter=args.length_filter,
            cot_source=args.cot_source,
            measure_final_answer=args.measure_final_answer,
            max_new_tokens=args.max_new_tokens,
            allow_no_baseline_cache=args.allow_no_baseline_cache,
            data_shard=data_shard,
        )

    if args.no_plot_after_run:
        sys.exit(0)

    plot_activation_patching(
        model_name,
        args.contrast,
        args.padded_dataset,
        subsets,
        args.scope,
        direction=args.direction,
        patch_positions=args.patch_positions,
        allow_truncated_span=args.allow_truncated_span,
        layers=args.layers,
        min_source_target_gap=args.min_source_target_gap,
        length_filter=args.length_filter,
        cot_source=args.cot_source,
    )
    plot_restore_disrupt_overlay(
        model_name,
        args.contrast,
        args.padded_dataset,
        subsets,
        args.scope,
        patch_positions=args.patch_positions,
        allow_truncated_span=args.allow_truncated_span,
        layers=args.layers,
        cot_source=args.cot_source,
        min_source_target_gap=args.min_source_target_gap,
        length_filter=args.length_filter,
    )
