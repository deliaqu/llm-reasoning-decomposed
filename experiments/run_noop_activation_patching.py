"""
Activation patching: noop direct correct -> noop direct wrong.

For each token-aligned (correct, wrong) pair from the same original GSM8K
instance, patch activations from the correct noop direct run into the wrong
noop direct run and measure how much the source-vs-target answer logprob
difference is restored at the final token position.

Usage:
    python run_noop_activation_patching.py --model_id meta-llama/Llama-3.3-70B-Instruct \\
        --scope layer
    python run_noop_activation_patching.py --model_id meta-llama/Llama-3.3-70B-Instruct \\
        --scope attn_head --layers 20 25 30 35 40
    python run_noop_activation_patching.py --plot_only
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList

from config import (
    COT_INSTRUCTION,
    DIRECT_INSTRUCTION,
    ANSWER_PREFIX,
    MODEL_NAME_MAP,
    NOOP_PATCHING_RESULT_DIR,
    P1_PADDED_DELTA0_PATCHING_RESULT_DIR,
    P1_PADDED_LT2_PATCHING_RESULT_DIR,
    GSM_SYM_WITHIN_TEMPLATE_PATCHING_RESULT_DIR,
)
from run_cot_swap_logit_lens import _style_layer_axis
from utils.activation_patching_utils import make_patching_hook
from utils.noop_utils import load_model, parse_symbol_bindings
from utils.shared_utils import remove_all_hooks

NOOP_EVAL_DIR = (
    Path(__file__).parent
    / "results" / "disentangled_evaluation" / "gsm_noop"
)
NOOP_CLEAN_EVAL_DIR = (
    Path(__file__).parent
    / "results" / "disentangled_evaluation" / "gsm_noop_clean"
)
GSM_SYMBOLIC_EVAL_DIR = (
    Path(__file__).parent
    / "results" / "disentangled_evaluation" / "transformers_direct" / "gsm_symbolic"
)
RESULT_DIR = Path(NOOP_PATCHING_RESULT_DIR)
PAIR_DATASET_PATHS = {
    "noop_clean_self": {},
    "noop_clean_self_ww": {},
    "gsm_sym_within_template": {},
    "filler_vs_noop": {
        "direct": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_filler_vs_noop_original_direct_pairs_filler_correct_noop_wrong.csv"
        ),
        "cot_boundary": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_filler_vs_noop_original_cot_pairs_filler_correct_noop_wrong.csv"
        ),
    },
    "filler_vs_noop_tfm": {
        "direct": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_filler_vs_noop_tfm_direct_pairs_filler_correct_noop_wrong_Llama-3.3-70B-Instruct.csv"
        ),
        "cot_boundary": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_filler_vs_noop_tfm_cot_pairs_filler_correct_noop_wrong_Llama-3.3-70B-Instruct.csv"
        ),
    },
    "filler_df_vs_noop_clean_tfm": {
        "direct": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_filler_df_vs_noop_clean_tfm_direct_pairs_filler_correct_noop_wrong_Llama-3.3-70B-Instruct.csv"
        ),
        "cot_boundary": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_filler_df_vs_noop_clean_tfm_cot_pairs_filler_correct_noop_wrong_Llama-3.3-70B-Instruct.csv"
        ),
    },
    "p1_vs_padded_lt2_tfm": {
        "direct": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_p1_vs_padded_lt2_tfm_pairs_orig_cot_correct_Llama-3.3-70B-Instruct.csv"
        ),
    },
    "p1_vs_padded_delta0_tfm": {
        "direct": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_p1_vs_padded_delta0_tfm_pairs_orig_cot_correct_Llama-3.3-70B-Instruct.csv"
        ),
    },
    # Entity-aligned padded variant: pairs P1 against
    # gsm_padded_symbolic_p1_aligned (same numeric instantiation as gsm_p1 by
    # construction). Matches the pair set the cot_swap activation-patching jobs
    # use, modulo the sym-CoT-correct filter (cot_boundary doesn't need a donor
    # CoT).
    "p1_vs_padded_aligned_delta0_tfm": {
        "direct": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_p1_vs_padded_aligned_delta0_tfm_pairs_orig_cot_correct_Llama-3.3-70B-Instruct.csv"
        ),
        # Same CSV; cot_boundary uses the same (question_c, question_w) pair
        # with CoT-template prompts and no donor CoT.
        "cot_boundary": (
            Path(__file__).parent
            / "results"
            / "disentangled_evaluation"
            / "gsm_p1_vs_padded_aligned_delta0_tfm_pairs_orig_cot_correct_Llama-3.3-70B-Instruct.csv"
        ),
    },
}

_DIRECTION_ALIASES = {
    "correct_to_wrong": "correct_to_wrong",
    "wrong_to_correct": "wrong_to_correct",
    "p1_to_padded": "correct_to_wrong",
    "padded_to_p1": "wrong_to_correct",
    "source_to_target": "correct_to_wrong",
    "target_to_source": "wrong_to_correct",
}


class AnswerBoundaryStoppingCriteria(StoppingCriteria):
    """Stop once the numeric answer after the final `####` appears complete."""

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        continuation_ids = input_ids[0, self.prompt_len:].detach().cpu().tolist()
        boundary_end = _boundary_end_index(self.tokenizer, continuation_ids)
        if boundary_end is None:
            return False

        answer_text = self.tokenizer.decode(
            continuation_ids[boundary_end:], skip_special_tokens=True
        ).strip()
        if not answer_text:
            return False

        match = re.match(r"^-?\d[\d,]*(?:\.\d+)?", answer_text)
        if match is None:
            return False

        return len(match.group(0)) < len(answer_text)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _apply_direct_template(text, tokenizer):
    messages = [{"role": "user", "content": f"{text}\n{DIRECT_INSTRUCTION}"}]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return base + ANSWER_PREFIX


def _apply_cot_template(text, tokenizer):
    messages = [{"role": "user", "content": f"{text}\n{COT_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _extract_question_span(prompt, question, tokenizer):
    """Inclusive token span covering `question` within `prompt`."""
    char_start = prompt.index(question)
    char_end = char_start + len(question)

    try:
        enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        start_candidates = [
            i for i, (s, e) in enumerate(offsets) if s <= char_start < e
        ]
        if start_candidates:
            q_start = start_candidates[0]
        else:
            q_start = min(
                i for i, (s, e) in enumerate(offsets) if s >= char_start and e > s
            )
        end_candidates = [
            i for i, (s, e) in enumerate(offsets) if s < char_end <= e
        ]
        if end_candidates:
            q_end = end_candidates[0]
        else:
            q_end = max(
                i for i, (s, e) in enumerate(offsets) if e <= char_end and e > s
            )
    except (TypeError, NotImplementedError):
        q_start = len(
            tokenizer(prompt[:char_start], add_special_tokens=False)["input_ids"]
        )
        q_end = (
            len(tokenizer(prompt[:char_end], add_special_tokens=False)["input_ids"]) - 1
        )
    return q_start, q_end


def _apply_cot_template_with_question_span(text, tokenizer):
    """Return CoT prompt plus inclusive token span covering the question text."""
    question = str(text)
    messages = [{"role": "user", "content": f"{question}\n{COT_INSTRUCTION}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    q_start, q_end = _extract_question_span(prompt, question, tokenizer)
    return prompt, q_start, q_end


def _apply_direct_template_with_question_span(text, tokenizer):
    """Return direct prompt plus inclusive token span covering the question text.

    Matches the prompt built by `_apply_direct_template` so token indices align
    with the prompts already stored in pair DataFrames.
    """
    question = str(text)
    messages = [{"role": "user", "content": f"{question}\n{DIRECT_INSTRUCTION}"}]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt = base + ANSWER_PREFIX
    q_start, q_end = _extract_question_span(prompt, question, tokenizer)
    return prompt, q_start, q_end


def _tok_len(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _coerce_bool(series):
    normalized = series.map(
        lambda x: x.strip().lower() if isinstance(x, str) else x
    )
    return normalized.map({
        True: True, False: False,
        1: True, 0: False, 1.0: True, 0.0: False,
        "true": True, "false": False,
        "1": True, "0": False,
    }).astype("boolean")


def _log_filter(context, before, after, reason):
    discarded = before - after
    print(f"{context}: kept {after}/{before}; discarded {discarded} ({reason})")


def load_noop_direct(model_id):
    # gsm_noop filenames use the bare model name after the last '/'
    model_suffix = model_id.split("/")[-1]
    files = sorted(NOOP_EVAL_DIR.glob(f"gsm_noop_set_*_results_{model_suffix}.csv"))
    if not files:
        raise FileNotFoundError(
            f"No gsm_noop direct files found for {model_suffix} in {NOOP_EVAL_DIR}"
        )
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["split"] = int(str(f).split("_set_")[1].split("_")[0])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["original_direct_correctness"] = _coerce_bool(df["original_direct_correctness"])
    before = len(df)
    df = df[df["original_direct_correctness"].notna()].copy()
    _log_filter("load_noop_direct", before, len(df), "missing original_direct_correctness")
    return df


def load_noop_clean_direct(model_id):
    model_suffix = model_id.split("/")[-1]
    files = sorted(
        NOOP_CLEAN_EVAL_DIR.glob(f"gsm_noop_clean_set_*_results_{model_suffix}.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No gsm_noop_clean direct files found for {model_suffix} in {NOOP_CLEAN_EVAL_DIR}"
        )
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["split"] = int(str(f).split("_set_")[1].split("_")[0])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["original_direct_correctness"] = _coerce_bool(df["original_direct_correctness"])
    before = len(df)
    df = df[df["original_direct_correctness"].notna()].copy()
    _log_filter("load_noop_clean_direct", before, len(df), "missing original_direct_correctness")
    return df


def load_noop_cot(model_id):
    df = load_noop_direct(model_id)
    df["original_cot_correctness"] = _coerce_bool(df["original_cot_correctness"])
    before = len(df)
    df = df[df["original_cot_correctness"].notna()].copy()
    _log_filter("load_noop_cot", before, len(df), "missing original_cot_correctness")
    return df


def load_noop_clean_cot(model_id):
    df = load_noop_clean_direct(model_id)
    df["original_cot_correctness"] = _coerce_bool(df["original_cot_correctness"])
    before = len(df)
    df = df[df["original_cot_correctness"].notna()].copy()
    _log_filter("load_noop_clean_cot", before, len(df), "missing original_cot_correctness")
    return df


def load_gsm_symbolic_results(model_id):
    model_suffix = model_id.split("/")[-1]
    transformers_suffix = f"{model_suffix}_transformers"
    files = sorted(
        GSM_SYMBOLIC_EVAL_DIR.glob(f"gsm_symbolic_set_*_results_{transformers_suffix}.csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No transformers-direct gsm_symbolic files found for {transformers_suffix} in {GSM_SYMBOLIC_EVAL_DIR}"
        )
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["split"] = int(str(f).split("_set_")[1].split("_")[0])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_gsm_symbolic_direct(model_id):
    df = load_gsm_symbolic_results(model_id)
    df["original_direct_correctness"] = _coerce_bool(df["original_direct_correctness"])
    before = len(df)
    df = df[df["original_direct_correctness"].notna()].copy()
    _log_filter("load_gsm_symbolic_direct", before, len(df), "missing original_direct_correctness")
    return df


def load_gsm_symbolic_cot(model_id):
    df = load_gsm_symbolic_results(model_id)
    df["original_cot_correctness"] = _coerce_bool(df["original_cot_correctness"])
    before = len(df)
    df = df[df["original_cot_correctness"].notna()].copy()
    _log_filter("load_gsm_symbolic_cot", before, len(df), "missing original_cot_correctness")
    return df


def _validate_pair_dataset(pair_dataset, experiment):
    if pair_dataset not in PAIR_DATASET_PATHS:
        raise ValueError(f"Unknown pair dataset: {pair_dataset}")
    dynamic_pair_datasets = {"noop_clean_self", "noop_clean_self_ww", "gsm_sym_within_template"}
    if pair_dataset not in dynamic_pair_datasets and experiment not in PAIR_DATASET_PATHS[pair_dataset]:
        raise ValueError(
            f"`{pair_dataset}` does not define a pair file for experiment `{experiment}`."
        )


def _is_p1_padded_dataset(pair_dataset):
    return pair_dataset in {
        "p1_vs_padded_lt2_tfm",
        "p1_vs_padded_delta0_tfm",
        "p1_vs_padded_aligned_delta0_tfm",
    }


def _is_gsm_sym_within_template_dataset(pair_dataset):
    return pair_dataset == "gsm_sym_within_template"


def _is_noop_clean_self_ww_dataset(pair_dataset):
    return pair_dataset == "noop_clean_self_ww"


def _uses_source_answer_as_gold(pair_dataset):
    """Within-template pair datasets (clean gsm_sym and wrong-wrong noop_clean)
    measure 'does the target adopt source's answer'. The gold passed into the
    cot_boundary metric must therefore be source's answer field, not target's.
    """
    return (
        _is_gsm_sym_within_template_dataset(pair_dataset)
        or _is_noop_clean_self_ww_dataset(pair_dataset)
    )


def _result_dir(pair_dataset):
    if pair_dataset == "p1_vs_padded_lt2_tfm":
        return Path(P1_PADDED_LT2_PATCHING_RESULT_DIR)
    if pair_dataset == "p1_vs_padded_delta0_tfm":
        return Path(P1_PADDED_DELTA0_PATCHING_RESULT_DIR)
    if pair_dataset == "p1_vs_padded_aligned_delta0_tfm":
        # Share the delta0 root but keep the pair_dataset directory name
        # distinct so legacy and aligned results don't collide.
        return Path(P1_PADDED_DELTA0_PATCHING_RESULT_DIR)
    if _is_gsm_sym_within_template_dataset(pair_dataset):
        return Path(GSM_SYM_WITHIN_TEMPLATE_PATCHING_RESULT_DIR)
    return RESULT_DIR


def _direction_filename_stem(dir_tag, pair_dataset):
    """Map the internal `dir_tag` to a directional filename stem used on disk.

    For noop datasets we now spell out the direction (`correct_to_wrong` /
    `wrong_to_correct`) instead of using an empty tag for the default direction.
    """
    canonical = dir_tag.lstrip("_")
    if _is_gsm_sym_within_template_dataset(pair_dataset) or _is_noop_clean_self_ww_dataset(pair_dataset):
        # Within-template pair datasets carry no correct/wrong semantics for the
        # direction; relabel so the on-disk filenames are interpretable.
        if canonical in ("correct_to_wrong", ""):
            return "source_to_target"
        if canonical == "wrong_to_correct":
            return "target_to_source"
        return canonical or "source_to_target"
    if _is_p1_padded_dataset(pair_dataset):
        return canonical or "p1_to_padded"
    if canonical == "":
        return "correct_to_wrong"
    if canonical == "w2c":
        return "wrong_to_correct"
    return canonical


def _result_path(model_name, experiment, scope, span_tag, layers_tag, dir_tag,
                 pair_dataset, extra_tag="", suffix=".jsonl"):
    """Build a result-file path.

    Hierarchical layout that mirrors `results/cot_swap_activation_patching/`:

        {result_dir}/{model}/[{pair_dataset}/]{experiment}/
            [{patch_positions}/]{scope}/{direction}{_lN}{_truncated?}{extra}{suffix}

    For `gsm_sym_within_template` the pair-dataset segment is dropped because
    that dataset has its own top-level result directory.
    """
    result_dir = _result_dir(pair_dataset)
    direction = _direction_filename_stem(dir_tag, pair_dataset)
    truncated = "_truncated" if span_tag.endswith("_truncated") else ""
    parts = [result_dir, model_name]
    if not _is_gsm_sym_within_template_dataset(pair_dataset):
        parts.append(pair_dataset)
    parts.append(experiment)
    if experiment == "cot_boundary":
        span_dir = (
            span_tag.replace("_truncated", "").lstrip("_") or "question_end"
        )
        parts.append(span_dir)
    elif experiment == "direct":
        span_dir = span_tag.replace("_truncated", "").lstrip("_")
        if span_dir:
            parts.append(span_dir)
    parts.append(scope)
    filename = f"{direction}{layers_tag}{truncated}{extra_tag}{suffix}"
    return Path(*parts) / filename


def _baselines_path(model_name, experiment, pair_dataset, dir_tag, span_tag=""):
    """Path to the cached unpatched-baseline jsonl for `cot_boundary` patching.

    Sits one directory above the per-scope result files so all four scopes
    (layer/mlp/attn_output/attn_head) share the same baselines for a given
    (experiment, span, direction):

        <result_dir>/<model>/[<pair_dataset>/]<experiment>/[<span_dir>/]
            baselines_<direction>[_truncated].jsonl
    """
    result_dir = _result_dir(pair_dataset)
    direction = _direction_filename_stem(dir_tag, pair_dataset)
    truncated = "_truncated" if span_tag.endswith("_truncated") else ""
    parts = [result_dir, model_name]
    if not _is_gsm_sym_within_template_dataset(pair_dataset):
        parts.append(pair_dataset)
    parts.append(experiment)
    if experiment == "cot_boundary":
        span_dir = (
            span_tag.replace("_truncated", "").lstrip("_") or "question_end"
        )
        parts.append(span_dir)
    return Path(*parts) / f"baselines_{direction}{truncated}.jsonl"


def _load_cot_boundary_baselines(baselines_path):
    """Read cached unpatched baselines into a dict keyed by pair tuple.

    Reads both the canonical `<baselines>.jsonl` and any sibling shard files
    `<baselines>.shard*.jsonl` produced by parallel `--baseline_shard` runs.
    Missing/empty files -> empty contribution. Corrupt/torn lines are skipped.
    Later records overwrite earlier ones if duplicate keys appear.
    """
    cache = {}
    path = Path(baselines_path)
    shard_pattern = f"{path.stem}.shard*{path.suffix}"
    files = sorted(path.parent.glob(shard_pattern)) if path.parent.exists() else []
    if path.exists() and path.stat().st_size > 0:
        files.insert(0, path)
    for f_path in files:
        if not f_path.exists() or f_path.stat().st_size == 0:
            continue
        with open(f_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (
                        int(rec["original_id"]),
                        int(rec["instance_c"]),
                        int(rec["instance_w"]),
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                cache[key] = rec
    return cache


def _scope_compare_path(model_name, experiment, layers_tag, filter_tag,
                        aggregation_tag, pair_dataset):
    result_dir = _result_dir(pair_dataset)
    parts = [result_dir, model_name]
    if not _is_gsm_sym_within_template_dataset(pair_dataset):
        parts.append(pair_dataset)
    parts.append(experiment)
    return (
        Path(*parts)
        / f"scope_compare{layers_tag}{filter_tag}{aggregation_tag}.png"
    )


def _canonical_direction(direction):
    try:
        return _DIRECTION_ALIASES[direction]
    except KeyError as exc:
        raise ValueError(f"Unknown direction: {direction}") from exc


def _direction_tag(direction, pair_dataset):
    canonical = _canonical_direction(direction)
    if _is_p1_padded_dataset(pair_dataset):
        return "_p1_to_padded" if canonical == "correct_to_wrong" else "_padded_to_p1"
    if _is_gsm_sym_within_template_dataset(pair_dataset) or _is_noop_clean_self_ww_dataset(pair_dataset):
        return "_source_to_target" if canonical == "correct_to_wrong" else "_target_to_source"
    return "" if canonical == "correct_to_wrong" else "_w2c"


def _direction_title(direction, pair_dataset):
    canonical = _canonical_direction(direction)
    if _is_p1_padded_dataset(pair_dataset):
        padded_label = "padded delta0" if pair_dataset == "p1_vs_padded_delta0_tfm" else "padded LT2"
        return f"Patch P1 into {padded_label}" if canonical == "correct_to_wrong" else f"Patch {padded_label} into P1"
    if _is_gsm_sym_within_template_dataset(pair_dataset) or _is_noop_clean_self_ww_dataset(pair_dataset):
        return (
            "Patch source instance into target instance"
            if canonical == "correct_to_wrong"
            else "Patch target instance into source instance"
        )
    return (
        "Build-up of correct answer"
        if canonical == "correct_to_wrong"
        else "Corruption of correct answer"
    )


def _direct_logprob_denom(result):
    if "src_metric" in result and "tgt_metric" in result:
        return float(result["src_metric"]) - float(result["tgt_metric"])
    src = float(result["src_lp"]["correct"]) - float(result["src_lp"]["wrong"])
    tgt = float(result["tgt_lp"]["correct"]) - float(result["tgt_lp"]["wrong"])
    return src - tgt


def _filter_by_direct_denom(results, min_abs_denom):
    if min_abs_denom is None or min_abs_denom <= 0:
        return results
    return [
        result for result in results
        if abs(_direct_logprob_denom(result)) > min_abs_denom
    ]


def _filter_by_cot_boundary_gap(results, min_abs_denom):
    if min_abs_denom is None or min_abs_denom <= 0:
        return results
    return [
        result for result in results
        if abs(float(result.get("source_target_gap", 0.0))) > min_abs_denom
    ]


def _denom_filter_tag(experiment, min_abs_denom):
    if min_abs_denom is None or min_abs_denom <= 0:
        return ""
    if experiment not in ("direct", "cot_boundary"):
        return ""
    return f"_absden_gt_{str(min_abs_denom).replace('.', 'p')}"


def _default_aggregate_by(pair_dataset):
    # Default to template-mean-of-means across all pair_datasets to avoid heavy
    # templates dominating the pair-level mean and to give template-level CIs.
    return "original_id"


def _aggregation_tag(aggregate_by, pair_dataset):
    if aggregate_by == "original_id":
        return "_template_avg"
    return ""


def _aggregate_effects(results, aggregate_by="row", key="effects"):
    if aggregate_by == "row":
        return np.array([r[key] for r in results], dtype=float), len(results), len(results)
    if aggregate_by != "original_id":
        raise ValueError(f"Unknown aggregate_by: {aggregate_by}")

    template_effects = []
    for original_id in sorted({int(r["original_id"]) for r in results}):
        group = [r for r in results if int(r["original_id"]) == original_id]
        group_effects = np.array([r[key] for r in group], dtype=float)
        template_effects.append(np.nanmean(group_effects, axis=0))
    return np.stack(template_effects, axis=0), len(template_effects), len(results)


def _bootstrap_ci(units, ci=95, n_resamples=1000, seed=0):
    """Bootstrap percentile CI for the mean along axis 0.

    `units` has shape (N, ...); we resample N units with replacement
    `n_resamples` times, take nanmean over the resampled axis, and
    return (lo, hi) percentile arrays with shape `units.shape[1:]`.
    """
    rng = np.random.default_rng(seed)
    n = units.shape[0]
    boot = np.empty((n_resamples,) + units.shape[1:], dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boot[i] = np.nanmean(units[idx], axis=0)
    lo_p = (100 - ci) / 2
    hi_p = 100 - lo_p
    return np.nanpercentile(boot, lo_p, axis=0), np.nanpercentile(boot, hi_p, axis=0)


def _load_external_pair_dataset(pair_dataset, experiment, model_id=None):
    """Load pair CSV. Paths in `PAIR_DATASET_PATHS` are templated on the
    Llama-3.3-70B-Instruct basename for historical reasons; if `model_id` names
    a different model, substitute its basename in the filename so per-model
    pair CSVs (built by `prepare_p1_padded_patching_pairs.py --model_name X`)
    are picked up automatically.
    """
    path = PAIR_DATASET_PATHS[pair_dataset][experiment]
    if model_id is not None:
        basename = model_id.split("/")[-1]
        if basename != "Llama-3.3-70B-Instruct" and \
                "Llama-3.3-70B-Instruct" in path.name:
            path = path.with_name(
                path.name.replace("Llama-3.3-70B-Instruct", basename)
            )
    if path is None or not path.exists():
        raise FileNotFoundError(f"Pair dataset not found: {path}")
    return pd.read_csv(path)


def build_gsm_sym_within_template_pairs(tokenizer, model_id, max_pairs_per_orig=None,
                                        seed=0):
    """
    Build same-template, different-instance GSM-Symbolic pairs.

    The source and target share the symbolic template (`original_id`) but have
    different concrete operand bindings and usually different numeric answers.
    We keep only direct-correct rows so the scored source/target answer tokens
    are behaviourally meaningful, then patch source activations into the target
    prompt to ask when source operand bindings start causally controlling the
    answer.
    """
    df = load_gsm_symbolic_direct(model_id)
    before_correct = len(df)
    df = df[df["original_direct_correctness"] == True].copy()
    _log_filter(
        "build_gsm_sym_within_template_pairs",
        before_correct,
        len(df),
        "non-direct-correct rows",
    )
    if df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    frames = []
    groups_total = 0
    groups_lt2 = 0
    groups_no_diff_answer = 0
    total_candidate_pairs = 0
    total_selected_pairs = 0
    for original_id, group in df.groupby("original_id", sort=True):
        groups_total += 1
        group = group.sort_values("instance").reset_index(drop=True)
        if len(group) < 2:
            groups_lt2 += 1
            continue

        candidate_pairs = [
            (i, j)
            for i in range(len(group))
            for j in range(i + 1, len(group))
            if str(group.loc[i, "answer"]).strip() != str(group.loc[j, "answer"]).strip()
        ]
        if not candidate_pairs:
            groups_no_diff_answer += 1
            continue
        total_candidate_pairs += len(candidate_pairs)

        if max_pairs_per_orig is None:
            # One deterministic nearby pair per source instance keeps the
            # default experiment linear in dataset size rather than quadratic.
            nearby_pairs = []
            for i in range(len(group) - 1):
                src_answer = str(group.loc[i, "answer"]).strip()
                for j in range(i + 1, len(group)):
                    if src_answer != str(group.loc[j, "answer"]).strip():
                        nearby_pairs.append((i, j))
                        break
            candidate_pairs = nearby_pairs or candidate_pairs[:1]
        elif len(candidate_pairs) > max_pairs_per_orig:
            chosen = rng.choice(len(candidate_pairs), size=max_pairs_per_orig, replace=False)
            candidate_pairs = [candidate_pairs[k] for k in sorted(chosen)]
        total_selected_pairs += len(candidate_pairs)

        rows = []
        for i, j in candidate_pairs:
            src = group.loc[i]
            tgt = group.loc[j]
            rows.append({
                "original_id": int(original_id),
                "instance_c": int(src["instance"]),
                "instance_w": int(tgt["instance"]),
                "id_c": int(src["id"]) if "id" in src else -1,
                "id_w": int(tgt["id"]) if "id" in tgt else -1,
                "question_c": src["question"],
                "question_w": tgt["question"],
                "symbolic_question_c": str(src.get("symbolic_question", "")),
                "symbolic_question_w": str(tgt.get("symbolic_question", "")),
                "answer_c": str(src["answer"]),
                "answer_w": str(tgt["answer"]),
                "patch_answer_c": str(src["answer"]),
                "patch_answer_w": str(tgt["answer"]),
                "original_direct_answer_c": str(src["original_direct_answer"]),
                "original_direct_answer_w": str(tgt["original_direct_answer"]),
                "symbol_binding_c": str(src.get("symbol_binding", "")),
                "symbol_binding_w": str(tgt.get("symbol_binding", "")),
                "numerical_abstraction_answer_c": str(src.get("numerical_abstraction_answer", "")),
                "numerical_abstraction_answer_w": str(tgt.get("numerical_abstraction_answer", "")),
                "split_c": int(src["split"]),
                "split_w": int(tgt["split"]),
            })
        frames.append(pd.DataFrame(rows))

    if not frames:
        print(
            "build_gsm_sym_within_template_pairs: no groups produced pairs; "
            f"groups_total={groups_total}, groups_lt2={groups_lt2}, "
            f"groups_no_diff_answer={groups_no_diff_answer}"
        )
        return pd.DataFrame()

    pairs = pd.concat(frames, ignore_index=True)
    print(
        "build_gsm_sym_within_template_pairs: "
        f"groups_total={groups_total}, groups_lt2={groups_lt2}, "
        f"groups_no_diff_answer={groups_no_diff_answer}, "
        f"candidate_pairs={total_candidate_pairs}, selected_pairs={total_selected_pairs}, "
        f"discarded_candidate_pairs={total_candidate_pairs - total_selected_pairs}"
    )
    pairs["full_prompt_c"] = pairs["question_c"].apply(
        lambda q: _apply_direct_template(q, tokenizer)
    )
    pairs["full_prompt_w"] = pairs["question_w"].apply(
        lambda q: _apply_direct_template(q, tokenizer)
    )
    pairs["tok_len_c"] = pairs["full_prompt_c"].apply(lambda t: _tok_len(t, tokenizer))
    pairs["tok_len_w"] = pairs["full_prompt_w"].apply(lambda t: _tok_len(t, tokenizer))
    pairs["tok_len"] = pairs["tok_len_c"]
    print(
        f"Built {len(pairs)} GSM-Symbolic within-template direct pairs "
        f"across {pairs['original_id'].nunique()} original_ids"
    )
    return pairs.sort_values(["original_id", "instance_c", "instance_w"]).reset_index(drop=True)


def build_gsm_sym_within_template_cot_pairs(model_id, max_pairs_per_orig=None, seed=0):
    """
    Build same-template, different-instance GSM-Symbolic pairs for CoT boundary
    patching. Both sides are CoT-correct so either answer can be used as the
    source-side override target for a given patch direction.
    """
    df = load_gsm_symbolic_cot(model_id)
    before_correct = len(df)
    df = df[df["original_cot_correctness"] == True].copy()
    _log_filter(
        "build_gsm_sym_within_template_cot_pairs",
        before_correct,
        len(df),
        "non-CoT-correct rows",
    )
    if df.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    frames = []
    groups_total = 0
    groups_lt2 = 0
    groups_no_diff_answer = 0
    total_candidate_pairs = 0
    total_selected_pairs = 0
    for original_id, group in df.groupby("original_id", sort=True):
        groups_total += 1
        group = group.sort_values("instance").reset_index(drop=True)
        if len(group) < 2:
            groups_lt2 += 1
            continue

        candidate_pairs = [
            (i, j)
            for i in range(len(group))
            for j in range(i + 1, len(group))
            if str(group.loc[i, "answer"]).strip() != str(group.loc[j, "answer"]).strip()
        ]
        if not candidate_pairs:
            groups_no_diff_answer += 1
            continue
        total_candidate_pairs += len(candidate_pairs)

        if max_pairs_per_orig is None:
            nearby_pairs = []
            for i in range(len(group) - 1):
                src_answer = str(group.loc[i, "answer"]).strip()
                for j in range(i + 1, len(group)):
                    if src_answer != str(group.loc[j, "answer"]).strip():
                        nearby_pairs.append((i, j))
                        break
            candidate_pairs = nearby_pairs or candidate_pairs[:1]
        elif len(candidate_pairs) > max_pairs_per_orig:
            chosen = rng.choice(len(candidate_pairs), size=max_pairs_per_orig, replace=False)
            candidate_pairs = [candidate_pairs[k] for k in sorted(chosen)]
        total_selected_pairs += len(candidate_pairs)

        rows = []
        for i, j in candidate_pairs:
            src = group.loc[i]
            tgt = group.loc[j]
            rows.append({
                "original_id": int(original_id),
                "instance_c": int(src["instance"]),
                "instance_w": int(tgt["instance"]),
                "id_c": int(src["id"]) if "id" in src else -1,
                "id_w": int(tgt["id"]) if "id" in tgt else -1,
                "question_c": src["question"],
                "question_w": tgt["question"],
                "symbolic_question_c": str(src.get("symbolic_question", "")),
                "symbolic_question_w": str(tgt.get("symbolic_question", "")),
                "answer_c": str(src["answer"]),
                "answer_w": str(tgt["answer"]),
                "original_cot_answer_c": str(src["original_cot_answer"]),
                "original_cot_answer_w": str(tgt["original_cot_answer"]),
                "original_cot_c": str(src.get("original_cot", "")),
                "original_cot_w": str(tgt.get("original_cot", "")),
                "symbol_binding_c": str(src.get("symbol_binding", "")),
                "symbol_binding_w": str(tgt.get("symbol_binding", "")),
                "numerical_abstraction_answer_c": str(src.get("numerical_abstraction_answer", "")),
                "numerical_abstraction_answer_w": str(tgt.get("numerical_abstraction_answer", "")),
                "split_c": int(src["split"]),
                "split_w": int(tgt["split"]),
            })
        frames.append(pd.DataFrame(rows))

    if not frames:
        print(
            "build_gsm_sym_within_template_cot_pairs: no groups produced pairs; "
            f"groups_total={groups_total}, groups_lt2={groups_lt2}, "
            f"groups_no_diff_answer={groups_no_diff_answer}"
        )
        return pd.DataFrame()

    pairs = pd.concat(frames, ignore_index=True)
    print(
        "build_gsm_sym_within_template_cot_pairs: "
        f"groups_total={groups_total}, groups_lt2={groups_lt2}, "
        f"groups_no_diff_answer={groups_no_diff_answer}, "
        f"candidate_pairs={total_candidate_pairs}, selected_pairs={total_selected_pairs}, "
        f"discarded_candidate_pairs={total_candidate_pairs - total_selected_pairs}"
    )
    print(
        f"Built {len(pairs)} GSM-Symbolic within-template CoT pairs "
        f"across {pairs['original_id'].nunique()} original_ids"
    )
    return pairs.sort_values(["original_id", "instance_c", "instance_w"]).reset_index(drop=True)


def _add_direct_question_spans(pairs, tokenizer):
    """Add q_start_c, q_end_c, q_start_w, q_end_w columns. Assumes pairs has
    question_c and question_w text columns."""
    span_c = pairs["question_c"].apply(
        lambda q: _apply_direct_template_with_question_span(q, tokenizer)
    )
    span_w = pairs["question_w"].apply(
        lambda q: _apply_direct_template_with_question_span(q, tokenizer)
    )
    pairs["q_start_c"] = span_c.apply(lambda t: int(t[1]))
    pairs["q_end_c"] = span_c.apply(lambda t: int(t[2]))
    pairs["q_start_w"] = span_w.apply(lambda t: int(t[1]))
    pairs["q_end_w"] = span_w.apply(lambda t: int(t[2]))
    return pairs


def build_aligned_direct_pairs(tokenizer, model_id, max_pairs_per_orig=None, seed=0,
                               pair_dataset="noop_clean_self", experiment="direct",
                               compute_question_spans=False):
    """
    Return DataFrame of token-aligned (correct, wrong) pairs.

    Source (correct): noop direct correct
    Target (wrong):   noop direct wrong

    Full prompt = direct template(noop question) + answer prefix.
    Pairs are filtered to equal full-prompt token length so that position -1
    corresponds to the same absolute position in both sequences.
    """
    if pair_dataset == "noop_clean_self":
        df = load_noop_clean_direct(model_id)

        correct = df[df["original_direct_correctness"] == True].copy()
        wrong   = df[df["original_direct_correctness"] == False].copy()
        print(
            "build_aligned_direct_pairs(noop_clean_self): "
            f"correct_rows={len(correct)}, wrong_rows={len(wrong)}, "
            f"discarded_non_boolean={len(df) - len(correct) - len(wrong)}"
        )

        for split_df in (correct, wrong):
            split_df["full_prompt"] = split_df["question"].apply(
                lambda q: _apply_direct_template(q, tokenizer)
            )
            split_df["tok_len"] = split_df["full_prompt"].apply(
                lambda t: _tok_len(t, tokenizer)
            )

        shared_ids = sorted(set(correct["original_id"]) & set(wrong["original_id"]))
        print(
            "build_aligned_direct_pairs(noop_clean_self): "
            f"correct_original_ids={correct['original_id'].nunique()}, "
            f"wrong_original_ids={wrong['original_id'].nunique()}, "
            f"shared_original_ids={len(shared_ids)}, "
            f"discarded_unshared_original_ids="
            f"{len(set(correct['original_id']) | set(wrong['original_id'])) - len(shared_ids)}"
        )

        frames = []
        total_merged_before_cap = 0
        total_merged_after_cap = 0
        groups_no_token_match = 0
        for oid in shared_ids:
            c = correct[correct["original_id"] == oid]
            w = wrong[wrong["original_id"] == oid]
            merged = c.merge(w, on=["original_id", "tok_len"], suffixes=("_c", "_w"))
            if merged.empty:
                groups_no_token_match += 1
            total_merged_before_cap += len(merged)
            if max_pairs_per_orig and len(merged) > max_pairs_per_orig:
                merged = merged.sample(max_pairs_per_orig, random_state=seed)
            total_merged_after_cap += len(merged)
            frames.append(merged)

        if not frames:
            return pd.DataFrame()

        pairs = pd.concat(frames, ignore_index=True)
        print(
            "build_aligned_direct_pairs(noop_clean_self): "
            f"groups_no_token_len_match={groups_no_token_match}, "
            f"candidate_pairs={total_merged_before_cap}, selected_pairs={total_merged_after_cap}, "
            f"discarded_pairs={total_merged_before_cap - total_merged_after_cap}"
        )
        print(
            f"Built {len(pairs)} token-aligned direct pairs across "
            f"{pairs['original_id'].nunique()} original_ids"
        )
        if compute_question_spans:
            # noop_clean_self merges on tok_len; question_c/_w arrive via merge
            # suffixes from the underlying `question` column.
            _add_direct_question_spans(pairs, tokenizer)
        return pairs

    if pair_dataset == "gsm_sym_within_template":
        pairs = build_gsm_sym_within_template_pairs(
            tokenizer, model_id, max_pairs_per_orig=max_pairs_per_orig, seed=seed
        )
        if compute_question_spans and not pairs.empty:
            _add_direct_question_spans(pairs, tokenizer)
        return pairs

    if pair_dataset == "noop_clean_self_ww":
        df = load_noop_clean_direct(model_id)

        wrong = df[df["original_direct_correctness"] == False].copy()
        print(
            "build_aligned_direct_pairs(noop_clean_self_ww): "
            f"wrong_rows={len(wrong)}, templates={wrong['original_id'].nunique()}"
        )
        if wrong.empty:
            return pd.DataFrame()

        wrong["full_prompt"] = wrong["question"].apply(
            lambda q: _apply_direct_template(q, tokenizer)
        )
        wrong["tok_len"] = wrong["full_prompt"].apply(
            lambda t: _tok_len(t, tokenizer)
        )

        frames = []
        total_candidate_pairs = 0
        total_distinct_answer_pairs = 0
        total_selected_pairs = 0
        rng = np.random.default_rng(seed)
        for oid, g in wrong.groupby("original_id", sort=True):
            if len(g) < 2:
                continue
            g = g.sort_values("instance").reset_index(drop=True)
            merged = g.merge(g, on=["original_id", "tok_len"], suffixes=("_c", "_w"))
            merged = merged[merged["instance_c"] < merged["instance_w"]]
            if merged.empty:
                continue
            total_candidate_pairs += len(merged)
            # Require distinct direct-mode wrong answers (analogous to clean
            # within-template's answer_c != answer_w filter).
            merged = merged[
                merged["original_direct_answer_c"].astype(str).str.strip()
                != merged["original_direct_answer_w"].astype(str).str.strip()
            ]
            if merged.empty:
                continue
            total_distinct_answer_pairs += len(merged)
            if max_pairs_per_orig and len(merged) > max_pairs_per_orig:
                merged = merged.sample(max_pairs_per_orig, random_state=int(rng.integers(0, 2**31 - 1)))
            total_selected_pairs += len(merged)
            frames.append(merged)

        if not frames:
            print("build_aligned_direct_pairs(noop_clean_self_ww): no pairs produced")
            return pd.DataFrame()

        pairs = pd.concat(frames, ignore_index=True)
        # Direct-mode within-template gold-selection (see
        # _uses_source_answer_as_gold + _score_columns_for_direction) reads
        # `patch_answer_*` (or `original_direct_answer_*`) as the value to
        # score at the readout. Both sides are noop-direct-wrong, so the
        # natural "value" is each instance's actual generated direct answer.
        pairs["answer_c"] = pairs["original_direct_answer_c"].astype(str)
        pairs["answer_w"] = pairs["original_direct_answer_w"].astype(str)
        pairs["patch_answer_c"] = pairs["original_direct_answer_c"].astype(str)
        pairs["patch_answer_w"] = pairs["original_direct_answer_w"].astype(str)
        print(
            "build_aligned_direct_pairs(noop_clean_self_ww): "
            f"candidate_pairs={total_candidate_pairs}, "
            f"distinct_answer_pairs={total_distinct_answer_pairs}, "
            f"selected_pairs={total_selected_pairs}, "
            f"discarded_same_answer={total_candidate_pairs - total_distinct_answer_pairs}, "
            f"discarded_by_cap={total_distinct_answer_pairs - total_selected_pairs}"
        )
        print(
            f"Built {len(pairs)} token-aligned direct pairs across "
            f"{pairs['original_id'].nunique()} original_ids"
        )
        if compute_question_spans:
            _add_direct_question_spans(pairs, tokenizer)
        return pairs.reset_index(drop=True)

    df = _load_external_pair_dataset(pair_dataset, experiment, model_id=model_id).copy()
    is_generic_pair_file = {"question_c", "question_w"}.issubset(df.columns)

    if is_generic_pair_file:
        pairs = df.copy()
        required = {"instance_c", "instance_w", "answer_c", "answer_w"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(
                f"Generic pair file for `{pair_dataset}` is missing columns: {sorted(missing)}"
            )
    else:
        n_before_len_filter = len(df)
        df["full_prompt_c"] = df["filler_question"].apply(
            lambda q: _apply_direct_template(q, tokenizer)
        )
        df["full_prompt_w"] = df["noop_question"].apply(
            lambda q: _apply_direct_template(q, tokenizer)
        )
        df["tok_len_c"] = df["full_prompt_c"].apply(lambda t: _tok_len(t, tokenizer))
        df["tok_len_w"] = df["full_prompt_w"].apply(lambda t: _tok_len(t, tokenizer))
        pairs = df[df["tok_len_c"] == df["tok_len_w"]].copy()
        _log_filter(
            f"build_aligned_direct_pairs({pair_dataset})",
            n_before_len_filter,
            len(pairs),
            "mismatched direct prompt token lengths",
        )
        pairs = pairs.rename(
            columns={
                "instance": "instance_c",
                "filler_question": "question_c",
                "noop_question": "question_w",
                "answer": "answer_c",
                "filler_model_answer": "original_direct_answer_c",
                "noop_model_answer": "original_direct_answer_w",
            }
        )
        pairs["instance_w"] = pairs["instance_c"]
        pairs["answer_w"] = pairs["answer_c"]

    pairs["full_prompt_c"] = pairs["question_c"].apply(
        lambda q: _apply_direct_template(q, tokenizer)
    )
    pairs["full_prompt_w"] = pairs["question_w"].apply(
        lambda q: _apply_direct_template(q, tokenizer)
    )
    pairs["tok_len_c"] = pairs["full_prompt_c"].apply(lambda t: _tok_len(t, tokenizer))
    pairs["tok_len_w"] = pairs["full_prompt_w"].apply(lambda t: _tok_len(t, tokenizer))
    pairs["tok_len"] = pairs["tok_len_c"]

    if pair_dataset == "p1_vs_padded_delta0_tfm":
        n_before_len_filter = len(pairs)
        pairs = pairs[pairs["tok_len_c"] == pairs["tok_len_w"]].copy()
        _log_filter(
            f"build_aligned_direct_pairs({pair_dataset})",
            n_before_len_filter,
            len(pairs),
            "mismatched final direct prompt token lengths",
        )

    if max_pairs_per_orig:
        n_before_cap = len(pairs)
        pairs = (
            pairs.sample(frac=1, random_state=seed)
            .groupby("original_id", group_keys=False, sort=False)
            .head(max_pairs_per_orig)
            .reset_index(drop=True)
        )
        _log_filter(
            f"build_aligned_direct_pairs({pair_dataset})",
            n_before_cap,
            len(pairs),
            f"max_pairs_per_orig={max_pairs_per_orig}",
        )
    print(
        f"Built {len(pairs)} direct pairs from `{pair_dataset}` "
        f"across {pairs['original_id'].nunique()} original_ids"
    )
    if compute_question_spans:
        _add_direct_question_spans(pairs, tokenizer)
    return pairs.reset_index(drop=True)


def build_cot_boundary_pairs(tokenizer, model_id, max_pairs_per_orig=None, seed=0,
                             pair_dataset="noop_clean_self", experiment="cot_boundary"):
    """
    Return (correct, wrong) pairs for CoT boundary patching.

    Source (correct): noop CoT correct
    Target (wrong):   noop CoT wrong

    Pairs are filtered to equal full-prompt token length under the CoT chat
    template so that position -1 corresponds to the same absolute index in
    both prompts. This avoids the positional-mismatch (rotary) noise that
    would otherwise arise when patching residuals between differently-indexed
    last-token positions.
    """
    if pair_dataset == "noop_clean_self":
        df = load_noop_clean_cot(model_id)

        correct = df[df["original_cot_correctness"] == True].copy()
        wrong   = df[df["original_cot_correctness"] == False].copy()
        print(
            "build_cot_boundary_pairs(noop_clean_self): "
            f"correct_rows={len(correct)}, wrong_rows={len(wrong)}, "
            f"discarded_non_boolean={len(df) - len(correct) - len(wrong)}"
        )

        for split_df in (correct, wrong):
            split_df["full_prompt"] = split_df["question"].apply(
                lambda q: _apply_cot_template(q, tokenizer)
            )
            split_df["tok_len"] = split_df["full_prompt"].apply(
                lambda t: _tok_len(t, tokenizer)
            )

        shared_ids = sorted(set(correct["original_id"]) & set(wrong["original_id"]))
        print(
            "build_cot_boundary_pairs(noop_clean_self): "
            f"correct_original_ids={correct['original_id'].nunique()}, "
            f"wrong_original_ids={wrong['original_id'].nunique()}, "
            f"shared_original_ids={len(shared_ids)}, "
            f"discarded_unshared_original_ids="
            f"{len(set(correct['original_id']) | set(wrong['original_id'])) - len(shared_ids)}"
        )

        frames = []
        total_merged_before_cap = 0
        total_merged_after_cap = 0
        groups_no_token_match = 0
        for oid in shared_ids:
            c = correct[correct["original_id"] == oid]
            w = wrong[wrong["original_id"] == oid]
            merged = c.merge(w, on=["original_id", "tok_len"], suffixes=("_c", "_w"))
            if merged.empty:
                groups_no_token_match += 1
            total_merged_before_cap += len(merged)
            if max_pairs_per_orig and len(merged) > max_pairs_per_orig:
                merged = merged.sample(max_pairs_per_orig, random_state=seed)
            total_merged_after_cap += len(merged)
            frames.append(merged)

        if not frames:
            return pd.DataFrame()

        pairs = pd.concat(frames, ignore_index=True)
        print(
            "build_cot_boundary_pairs(noop_clean_self): "
            f"candidate_pairs={total_merged_before_cap}, selected_pairs={total_merged_after_cap}, "
            f"discarded_pairs={total_merged_before_cap - total_merged_after_cap}"
        )
        print(
            f"Built {len(pairs)} CoT boundary pairs across "
            f"{pairs['original_id'].nunique()} original_ids"
        )
        return pairs

    if pair_dataset == "noop_clean_self_ww":
        df = load_noop_clean_cot(model_id)

        wrong = df[df["original_cot_correctness"] == False].copy()
        print(
            "build_cot_boundary_pairs(noop_clean_self_ww): "
            f"wrong_rows={len(wrong)}, templates={wrong['original_id'].nunique()}"
        )

        wrong["full_prompt"] = wrong["question"].apply(
            lambda q: _apply_cot_template(q, tokenizer)
        )
        wrong["tok_len"] = wrong["full_prompt"].apply(
            lambda t: _tok_len(t, tokenizer)
        )

        frames = []
        total_candidate_pairs = 0
        total_distinct_answer_pairs = 0
        total_selected_pairs = 0
        groups_no_pairs = 0
        rng = np.random.default_rng(seed)
        for oid, g in wrong.groupby("original_id", sort=True):
            if len(g) < 2:
                continue
            g = g.sort_values("instance").reset_index(drop=True)
            # Self-merge on tok_len, then keep only i<j unordered pairs.
            merged = g.merge(g, on=["original_id", "tok_len"], suffixes=("_c", "_w"))
            merged = merged[merged["instance_c"] < merged["instance_w"]]
            if merged.empty:
                groups_no_pairs += 1
                continue
            total_candidate_pairs += len(merged)
            # Require distinct wrong CoT answers: if both prompts produced the
            # same wrong number, the binding-transfer measurement is degenerate
            # (target stays at the same number whether patched or not). Mirrors
            # the `answer_c != answer_w` filter in build_gsm_sym_within_template_cot_pairs.
            merged = merged[
                merged["original_cot_answer_c"].astype(str).str.strip()
                != merged["original_cot_answer_w"].astype(str).str.strip()
            ]
            if merged.empty:
                groups_no_pairs += 1
                continue
            total_distinct_answer_pairs += len(merged)
            if max_pairs_per_orig and len(merged) > max_pairs_per_orig:
                merged = merged.sample(max_pairs_per_orig, random_state=int(rng.integers(0, 2**31 - 1)))
            total_selected_pairs += len(merged)
            frames.append(merged)

        if not frames:
            print(
                "build_cot_boundary_pairs(noop_clean_self_ww): no groups produced "
                f"pairs; groups_no_pairs={groups_no_pairs}"
            )
            return pd.DataFrame()

        pairs = pd.concat(frames, ignore_index=True)
        # Within-template gold-selection (see _uses_source_answer_as_gold) reads
        # `answer_*` as the value to track in the target. Both sides are
        # noop-wrong, so the natural "value" is each instance's actual generated
        # CoT answer (the wrong number the model produced). Mirror it into the
        # gold-style column so the downstream runner picks it up.
        pairs["answer_c"] = pairs["original_cot_answer_c"].astype(str)
        pairs["answer_w"] = pairs["original_cot_answer_w"].astype(str)
        pairs["original_cot_answer_c"] = pairs["original_cot_answer_c"].astype(str)
        pairs["original_cot_answer_w"] = pairs["original_cot_answer_w"].astype(str)
        print(
            "build_cot_boundary_pairs(noop_clean_self_ww): "
            f"candidate_pairs={total_candidate_pairs}, "
            f"distinct_answer_pairs={total_distinct_answer_pairs}, "
            f"selected_pairs={total_selected_pairs}, "
            f"discarded_same_answer={total_candidate_pairs - total_distinct_answer_pairs}, "
            f"discarded_by_cap={total_distinct_answer_pairs - total_selected_pairs}"
        )
        print(
            f"Built {len(pairs)} CoT boundary pairs across "
            f"{pairs['original_id'].nunique()} original_ids"
        )
        return pairs.reset_index(drop=True)

    if pair_dataset == "gsm_sym_within_template":
        return build_gsm_sym_within_template_cot_pairs(
            model_id, max_pairs_per_orig=max_pairs_per_orig, seed=seed
        )

    df = _load_external_pair_dataset(pair_dataset, experiment, model_id=model_id).copy()

    # P1-vs-padded pair CSVs use a different schema from the filler/noop CSVs:
    # columns are already `question_c`/`question_w`/`answer_c`/`answer_w`/
    # `instance_c`/`instance_w` (built by prepare_p1_padded_patching_pairs.py).
    # Branch here so we don't try to rename non-existent `filler_question`.
    p1_padded_schema = _is_p1_padded_dataset(pair_dataset)

    n_before_len_filter = len(df)
    src_col = "question_c" if p1_padded_schema else "filler_question"
    tgt_col = "question_w" if p1_padded_schema else "noop_question"
    df["full_prompt_c"] = df[src_col].apply(
        lambda q: _apply_cot_template(q, tokenizer)
    )
    df["full_prompt_w"] = df[tgt_col].apply(
        lambda q: _apply_cot_template(q, tokenizer)
    )
    df["tok_len_c"] = df["full_prompt_c"].apply(lambda t: _tok_len(t, tokenizer))
    df["tok_len_w"] = df["full_prompt_w"].apply(lambda t: _tok_len(t, tokenizer))
    df = df[df["tok_len_c"] == df["tok_len_w"]].copy()
    _log_filter(
        f"build_cot_boundary_pairs({pair_dataset})",
        n_before_len_filter,
        len(df),
        "mismatched cot prompt token lengths",
    )

    # Always shuffle the pair order with a fixed seed so partial completion
    # (a single layer-shard or run that doesn't finish all rows) gives an
    # unbiased random sample, not a low-original_id prefix. All 80 layer-shards
    # use the same fixed seed → consistent ordering across layers.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_pairs_per_orig:
        n_before_cap = len(df)
        df = (
            df.groupby("original_id", group_keys=False, sort=False)
            .head(max_pairs_per_orig)
            .reset_index(drop=True)
        )
        _log_filter(
            f"build_cot_boundary_pairs({pair_dataset})",
            n_before_cap,
            len(df),
            f"max_pairs_per_orig={max_pairs_per_orig}",
        )
    if p1_padded_schema:
        # P1 CSV already has the canonical (_c, _w) column names — no rename
        # needed. Crucially answer_c != answer_w (the whole point: p1 and
        # padded have different gold answers) so we must NOT collapse them
        # the way the filler_df branch does below.
        pairs = df
    else:
        pairs = df.rename(
            columns={
                "instance": "instance_c",
                "filler_question": "question_c",
                "noop_question": "question_w",
                "answer": "answer_c",
                "filler_model_answer": "original_cot_answer_c",
                "noop_model_answer": "original_cot_answer_w",
            }
        )
        pairs["instance_w"] = pairs["instance_c"]
        pairs["answer_w"] = pairs["answer_c"]
    print(
        f"Built {len(pairs)} CoT boundary pairs from `{pair_dataset}` "
        f"across {pairs['original_id'].nunique()} original_ids"
    )
    return pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

def _encode(text, tokenizer, device):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in ids.items()}


def _encode_batch(texts, tokenizer, device):
    ids = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=False)
    return {k: v.to(device) for k, v in ids.items()}


def _make_last_token_hook(state_dict, key):
    def save_last_token(module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        state_dict[key] = tensor[:, -1:, :].detach().clone()
    return save_last_token


def _make_last_token_input_hook(state_dict, key):
    def save_last_token_input(module, input, output):
        tensor = input[0]
        state_dict[key] = tensor[:, -1:, :].detach().clone()
    return save_last_token_input


def _make_index_saving_hook(state_dict, key, indices, save_input=False):
    def save_indices(module, input, output):
        tensor = input[0] if save_input else (output[0] if isinstance(output, tuple) else output)
        state_dict[key] = tensor[:, indices, :].detach().clone()
    return save_indices


def _register_last_token_saving_hooks(model, scope):
    """
    Register only the hooks needed for noop patching, and cache only the final
    prompt token because this script always patches position -1.
    """
    state_dict, state_hooks = {}, []
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        if scope == "layer":
            module = model.model.layers[layer_idx]
            key = f"layer_{layer_idx}"
        elif scope == "mlp":
            module = model.model.layers[layer_idx].mlp
            key = f"layer_mlp_{layer_idx}"
        elif scope == "attn_output":
            module = model.model.layers[layer_idx].self_attn
            key = f"layer_attn_output_{layer_idx}"
        elif scope == "attn_head":
            module = model.model.layers[layer_idx].self_attn.o_proj
            key = f"layer_attn_head_{layer_idx}"
        else:
            raise ValueError(f"Unknown scope: {scope}")

        if scope == "attn_head":
            state_hooks.append(module.register_forward_hook(_make_last_token_input_hook(state_dict, key)))
        else:
            state_hooks.append(module.register_forward_hook(_make_last_token_hook(state_dict, key)))

    return state_dict, state_hooks


def _register_index_saving_hooks(model, scope, indices, layers=None):
    """Register hooks for a selected token span at every active layer."""
    state_dict, state_hooks = {}, []
    num_layers = model.config.num_hidden_layers
    active_layers = layers if layers is not None else list(range(num_layers))

    for layer_idx in active_layers:
        if scope == "layer":
            module = model.model.layers[layer_idx]
            key = f"layer_{layer_idx}"
            save_input = False
        elif scope == "mlp":
            module = model.model.layers[layer_idx].mlp
            key = f"layer_mlp_{layer_idx}"
            save_input = False
        elif scope == "attn_output":
            module = model.model.layers[layer_idx].self_attn
            key = f"layer_attn_output_{layer_idx}"
            save_input = False
        elif scope == "attn_head":
            module = model.model.layers[layer_idx].self_attn.o_proj
            key = f"layer_attn_head_{layer_idx}"
            save_input = True
        else:
            raise ValueError(f"Unknown scope: {scope}")

        state_hooks.append(
            module.register_forward_hook(
                _make_index_saving_hook(state_dict, key, indices, save_input=save_input)
            )
        )

    return state_dict, state_hooks


def _literal_to_flexible_regex(text):
    """Escape literal text while allowing harmless whitespace differences."""
    return "".join(r"\s+" if ch.isspace() else re.escape(ch) for ch in str(text))


def _bound_value_regex(value):
    value = str(value).strip()
    choices = [value]
    if "," in value:
        choices.append(value.replace(",", ""))
    return "(?:" + "|".join(re.escape(v) for v in dict.fromkeys(choices)) + ")"


_SMALL_NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen",
}
_TENS_WORDS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}
_ORDINAL_DENOM_WORDS = {
    2: ("half", "halves"),
    3: ("third", "thirds"),
    4: ("fourth", "fourths"),
    5: ("fifth", "fifths"),
    6: ("sixth", "sixths"),
    7: ("seventh", "sevenths"),
    8: ("eighth", "eighths"),
    9: ("ninth", "ninths"),
    10: ("tenth", "tenths"),
    11: ("eleventh", "elevenths"),
    12: ("twelfth", "twelfths"),
}
_MULTIPLIER_WORDS = {
    1: ["once"],
    2: ["twice", "double"],
    3: ["thrice", "triple"],
    4: ["quadruple"],
    5: ["quintuple"],
    6: ["sextuple"],
}


def _int_to_words(n):
    """Return common lowercase English spellings for non-negative integers."""
    if n < 0 or n >= 10000:
        return []
    if n < 20:
        return [_SMALL_NUM_WORDS[n]]
    if n < 100:
        tens = (n // 10) * 10
        rem = n % 10
        if rem == 0:
            return [_TENS_WORDS[tens]]
        return [
            f"{_TENS_WORDS[tens]} {_SMALL_NUM_WORDS[rem]}",
            f"{_TENS_WORDS[tens]}-{_SMALL_NUM_WORDS[rem]}",
        ]
    if n < 1000:
        hundreds = n // 100
        rem = n % 100
        prefix = f"{_SMALL_NUM_WORDS[hundreds]} hundred"
        if rem == 0:
            return [prefix]
        return [f"{prefix} {suffix}" for suffix in _int_to_words(rem)]
    thousands = n // 1000
    rem = n % 1000
    prefix = f"{_int_to_words(thousands)[0]} thousand"
    if rem == 0:
        return [prefix]
    return [f"{prefix} {suffix}" for suffix in _int_to_words(rem)]


def _fraction_to_words(numerator, denominator):
    if numerator < 1 or denominator not in _ORDINAL_DENOM_WORDS:
        return []
    numerator_words = _int_to_words(numerator)
    if not numerator_words:
        return []
    singular, plural = _ORDINAL_DENOM_WORDS[denominator]
    denom_words = [(singular, plural)]
    if denominator == 4:
        denom_words.append(("quarter", "quarters"))
    forms = []
    for singular_word, plural_word in denom_words:
        denom_word = singular_word if numerator == 1 else plural_word
        for n_words in numerator_words:
            forms.append(f"{n_words}-{denom_word}")
            forms.append(f"{n_words} {denom_word}")
        if numerator == 1:
            article = "an" if singular_word[0] in "aeiou" else "a"
            forms.extend([singular_word, f"{article} {singular_word}"])
    if numerator == 1 and denominator == 2:
        forms.extend(["half", "a half", "one half"])
    return forms


def _value_surface_patterns(value):
    """Regex alternatives for a bound value as it may appear in a question."""
    raw = str(value).strip()
    choices = [raw]
    no_comma = raw.replace(",", "")
    choices.append(no_comma)
    try:
        number = float(no_comma)
        if number.is_integer():
            int_text = str(int(number))
            choices.append(int_text)
            choices.append(f"d{int_text}")
            choices.extend(_int_to_words(int(number)))
            if int(number) in _MULTIPLIER_WORDS:
                choices.extend(_MULTIPLIER_WORDS[int(number)])
    except (TypeError, ValueError):
        pass
    frac_match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", raw)
    if frac_match is not None:
        choices.extend(
            _fraction_to_words(
                int(frac_match.group(1)),
                int(frac_match.group(2)),
            )
        )
    patterns = []
    for choice in dict.fromkeys(c for c in choices if c):
        if re.fullmatch(r"[A-Za-z]+(?:[ -][A-Za-z]+)*", choice):
            patterns.append(rf"(?<![A-Za-z]){re.escape(choice)}(?![A-Za-z])")
        else:
            patterns.append(rf"(?<![\w.]){re.escape(choice)}(?![\w]|(?:\.\d))")
    return patterns


def _symbolic_variable_order(symbolic_question, bindings):
    if not isinstance(symbolic_question, str) or not symbolic_question.strip():
        return []
    var_names = sorted(bindings, key=len, reverse=True)
    if not var_names:
        return []
    var_re = re.compile(
        r"(?<![A-Za-z0-9_])("
        + "|".join(re.escape(v) for v in var_names)
        + r")(?![A-Za-z0-9_])"
    )
    return [match.group(1) for match in var_re.finditer(symbolic_question)]


def _value_appears_in_text(text, value):
    if not isinstance(text, str) or not text:
        return False
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in _value_surface_patterns(value)
    )


def _symbolic_value_char_spans(question, symbolic_question, bindings):
    """Use the symbolic template to locate concrete values by variable name."""
    if not isinstance(symbolic_question, str) or not symbolic_question.strip():
        return {}
    var_names = sorted(bindings, key=len, reverse=True)
    if not var_names:
        return {}

    var_re = re.compile(
        r"(?<![A-Za-z0-9_])("
        + "|".join(re.escape(v) for v in var_names)
        + r")(?![A-Za-z0-9_])"
    )
    parts = ["^"]
    group_to_var = {}
    last = 0
    group_idx = 0
    for match in var_re.finditer(symbolic_question):
        parts.append(_literal_to_flexible_regex(symbolic_question[last:match.start()]))
        var_name = match.group(1)
        group_name = f"v{group_idx}"
        group_idx += 1
        group_to_var[group_name] = var_name
        parts.append(f"(?P<{group_name}>{_bound_value_regex(bindings[var_name])})")
        last = match.end()
    parts.append(_literal_to_flexible_regex(symbolic_question[last:]))
    parts.append("$")

    match = re.match("".join(parts), str(question))
    if match is None:
        return {}

    spans = {}
    for group_name, var_name in group_to_var.items():
        char_span = match.span(group_name)
        if char_span[0] >= 0:
            spans.setdefault(var_name, []).append(char_span)
    return spans


def _sequential_symbolic_value_char_spans(question, symbolic_question, bindings):
    """Locate values by scanning concrete text in symbolic variable order.

    This is a fallback for GSM-Symbolic rows where the full symbolic template
    does not literally match the concrete question because entities, casing, or
    value spellings changed (e.g. `y = 6` -> "six times"). It still preserves
    variable identity for duplicated values by consuming occurrences in the
    order variables appear in the symbolic template.
    """
    spans = {}
    cursor = 0
    for var_name in _symbolic_variable_order(symbolic_question, bindings):
        patterns = _value_surface_patterns(bindings[var_name])
        best_start = None
        best_end = None
        for pattern in patterns:
            candidate = re.search(pattern, str(question)[cursor:], flags=re.IGNORECASE)
            if candidate is None:
                continue
            start = cursor + candidate.start()
            end = cursor + candidate.end()
            if best_start is None or start < best_start:
                best_start = start
                best_end = end
        if best_start is None:
            return {}
        span = (best_start, best_end)
        spans.setdefault(var_name, []).append(span)
        cursor = span[1]
    return spans


def _value_token_positions(prompt, question, symbol_binding, tokenizer,
                           symbolic_question=None):
    """Token positions in `prompt` covering each operand value, keyed by
    variable name.

    Restricts the search to the question substring (so a digit appearing
    elsewhere in the chat template doesn't get patched). Order within an
    operand follows tokenization order. Missing or unfindable operands
    map to an empty list; the alignment helper uses that as a filter
    signal.
    """
    bindings = parse_symbol_bindings(symbol_binding)
    if not bindings:
        return {}
    symbolic_spans = _symbolic_value_char_spans(
        question, symbolic_question, bindings
    )
    if not symbolic_spans and symbolic_question:
        symbolic_spans = _sequential_symbolic_value_char_spans(
            question, symbolic_question, bindings
        )
    symbolic_var_names = set(_symbolic_variable_order(symbolic_question, bindings))
    value_counts = {}
    for raw_value in bindings.values():
        value = str(raw_value).strip()
        norm_value = value.replace(",", "")
        value_counts[norm_value] = value_counts.get(norm_value, 0) + 1

    question_char_start = prompt.find(question)
    if question_char_start < 0:
        return {}

    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    result = {}
    for var_name, raw_value in bindings.items():
        value = str(raw_value).strip()
        if not value:
            result[var_name] = []
            continue
        if (
            var_name not in symbolic_var_names
            and (
                not _value_appears_in_text(symbolic_question, value)
                or var_name == "alphabets"
            )
        ):
            # Some bindings are implicit constants (e.g. alphabets = 26) and
            # have no surface token to patch. Omit them from the alignment
            # contract rather than forcing the pair to be skipped.
            continue

        local_spans = symbolic_spans.get(var_name)
        if local_spans is None:
            # Fallback only when the value is unique in the binding. If two
            # variables bind to the same surface value, plain `search` cannot
            # tell which occurrence belongs to which variable.
            if value_counts.get(value.replace(",", ""), 0) > 1:
                result[var_name] = []
                continue
            m = None
            for pattern in _value_surface_patterns(value):
                m = re.search(pattern, str(question), flags=re.IGNORECASE)
                if m is not None:
                    break
            if not m:
                result[var_name] = []
                continue
            local_spans = [m.span()]

        positions = []
        for local_start, local_end in local_spans:
            char_s = question_char_start + local_start
            char_e = question_char_start + local_end
            positions.extend(
                i for i, (s, e) in enumerate(offsets)
                if s != e and max(s, char_s) < min(e, char_e)
            )
        result[var_name] = positions
    return result


def _aligned_value_token_indices(src_positions, tgt_positions):
    """Align per-operand token positions by variable name.

    Returns (src_flat, tgt_flat) — flat lists where the i-th source position
    is to be copied to the i-th target position. Filters the pair (returns
    (None, None)) when any operand is missing on either side, has empty
    positions, or has mismatched token count source vs. target.
    """
    if not src_positions or not tgt_positions:
        return None, None
    if set(src_positions) != set(tgt_positions):
        return None, None
    src_flat, tgt_flat = [], []
    for var in sorted(src_positions):
        s_pos = src_positions[var]
        t_pos = tgt_positions[var]
        if not s_pos or not t_pos or len(s_pos) != len(t_pos):
            return None, None
        src_flat.extend(s_pos)
        tgt_flat.extend(t_pos)
    if not src_flat:
        return None, None
    return src_flat, tgt_flat


def _cot_patch_indices(source_start, source_end, target_start, target_end,
                       allow_truncated_span=False):
    source_len = source_end - source_start + 1
    target_len = target_end - target_start + 1
    if source_len != target_len and not allow_truncated_span:
        return [], []
    span_len = min(source_len, target_len)
    return (
        list(range(source_start, source_start + span_len)),
        list(range(target_start, target_start + span_len)),
    )


def _make_span_patching_hook(hs, target_indices, head_idx=None, n_heads=None):
    def patching_hook(module, input, output):
        original_output = output
        output_tensor = output[0] if isinstance(output, tuple) else output

        # During autoregressive decoding, only the new token is materialized.
        # The question span exists only in the initial prompt prefill.
        if output_tensor.size(1) <= max(target_indices):
            return original_output

        if head_idx is None:
            patched = output_tensor.clone()
            source_tensor = hs.to(patched.device)
            patched[:, target_indices, :] = source_tensor
            return (patched,) + original_output[1:] if isinstance(original_output, tuple) else patched

        input_tensor = input[0]
        if input_tensor.size(1) <= max(target_indices):
            return original_output

        device = module.weight.device
        input_tensor = input_tensor.to(device).clone()
        source_tensor = hs.to(device)
        batch_size, seq_len, hidden_size = input_tensor.shape
        head_dim = hidden_size // n_heads

        input_heads = input_tensor.view(batch_size, seq_len, n_heads, head_dim)
        source_heads = source_tensor.view(batch_size, len(target_indices), n_heads, head_dim)
        input_heads[:, target_indices, head_idx, :] = source_heads[:, :, head_idx, :]

        merged_input = input_heads.view(batch_size, seq_len, hidden_size)
        patched = merged_input @ module.weight.t()
        if module.bias is not None:
            patched = patched + module.bias
        return (patched,) + original_output[1:] if isinstance(original_output, tuple) else patched

    return patching_hook


def _extract_cot_answer_text(generation):
    parts = str(generation).split("####")
    if len(parts) >= 2:
        response = parts[-1].strip()
        if response:
            return response
        return parts[-2].strip()
    return str(generation).strip()


def _extract_last_number(text):
    text = re.sub(r"(\d),(\d)", r"\1\2", str(text))
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def _extract_emitted_answer(generation):
    """Return the numeric answer the model emitted *after* `####`, or None
    if the generation doesn't contain a well-formed answer marker."""
    text = str(generation)
    if "####" not in text:
        return None
    answer_text = text.split("####")[-1].strip()
    return _extract_last_number(answer_text)


def _numbers_equal(a, b):
    """String numeric equality with tolerance for trailing decimal zeros."""
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a).strip() == str(b).strip()


def _is_correct_final_answer(generation, reference):
    answer_text = _extract_cot_answer_text(generation)
    pred_num = _extract_last_number(answer_text)
    ref_num = _extract_last_number(reference)
    if pred_num is None or ref_num is None:
        return False
    try:
        return float(pred_num) == float(ref_num)
    except Exception:
        return pred_num == ref_num


def _greedy_generate(model, tokenizer, inputs, max_new_tokens, return_scores=False):
    prompt_len = inputs["input_ids"].shape[1]
    stopping_criteria = StoppingCriteriaList([
        AnswerBoundaryStoppingCriteria(tokenizer, prompt_len)
    ])
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stopping_criteria=stopping_criteria,
        return_dict_in_generate=return_scores,
        output_scores=return_scores,
    )
    sequences = output.sequences if return_scores else output
    continuation_ids = sequences[0, prompt_len:].detach().cpu().tolist()
    continuation_text = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    if not return_scores:
        return continuation_ids, continuation_text
    return continuation_ids, continuation_text, output.scores


def _find_last_subsequence(seq, pattern):
    if not pattern or len(pattern) > len(seq):
        return None
    for start in range(len(seq) - len(pattern), -1, -1):
        if seq[start:start + len(pattern)] == pattern:
            return start
    return None


# Module-level BPE-pattern cache. `_boundary_end_index` and the
# `AnswerBoundaryStoppingCriteria` together call this function thousands of
# times per pair (once per decode step × N_layers), and the underlying
# `tokenizer.encode("####")` runs the full BPE merge loop each time.
# Caching the encoded patterns per tokenizer turns those into dict lookups.
_HASH_BOUNDARY_PATTERN_CACHE = {}


def _hash_boundary_patterns(tokenizer):
    key = id(tokenizer)
    cached = _HASH_BOUNDARY_PATTERN_CACHE.get(key)
    if cached is None:
        cached = (
            tokenizer.encode("#### ", add_special_tokens=False),
            tokenizer.encode("####", add_special_tokens=False),
        )
        _HASH_BOUNDARY_PATTERN_CACHE[key] = cached
    return cached


def _boundary_end_index(tokenizer, continuation_ids):
    candidates = _hash_boundary_patterns(tokenizer)
    best_end = None
    for pattern in candidates:
        start = _find_last_subsequence(continuation_ids, pattern)
        if start is None:
            continue
        end = start + len(pattern)
        if best_end is None or end > best_end:
            best_end = end
    if best_end is None:
        return None

    while best_end < len(continuation_ids):
        piece = tokenizer.decode(
            [continuation_ids[best_end]], skip_special_tokens=False
        )
        if piece.strip():
            break
        best_end += 1
    return best_end


def _gold_answer_logprob_at_generated_boundary(
    model, tokenizer, prompt_ids, continuation_ids, gold_answer
):
    boundary_end = _boundary_end_index(tokenizer, continuation_ids)
    if boundary_end is None:
        return np.nan

    device = next(model.parameters()).device
    prefix_ids = prompt_ids + continuation_ids[:boundary_end]
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    gold_candidates = _gold_first_token_candidates(tokenizer, gold_answer)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
    # Robust to leading-whitespace BPE merging: take max over all plausible
    # first-content-token variants of the gold answer.
    return max(lp[tid].item() for tid in gold_candidates)


def _score_cot_boundary_from_cached_cot(
    model, tokenizer, prompt_text, cached_cot, gold_answers
):
    """Score `gold_answers` at the #### boundary of a precomputed (cached)
    CoT using a single forward pass over (prompt + cached_cot[:boundary]),
    skipping the autoregressive generation loop.

    Mathematically equivalent to greedy-decoding from `prompt_text` and
    reading boundary scores, *provided* the cached CoT was produced under
    the same chat-template + greedy-decode conventions used at runtime
    (transformers_direct convention in this project).

    Returns:
        dict[gold_answer -> (correct: bool, logprob: float)]
        — `correct` is whether `cached_cot` itself ends with that gold;
        `logprob` is log P of the gold's first content token at the
        boundary. If the cached CoT has no parseable #### boundary, every
        gold gets (False, nan).
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    cot_ids = tokenizer(cached_cot, add_special_tokens=False)["input_ids"]

    boundary_end = _boundary_end_index(tokenizer, cot_ids)
    if boundary_end is None:
        return {g: (False, float("nan")) for g in gold_answers}

    prefix_ids = list(prompt_ids) + list(cot_ids[:boundary_end])
    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]

    result = {}
    for gold in gold_answers:
        gold_candidates = _gold_first_token_candidates(tokenizer, gold)
        gold_lp = max(lp[tid].item() for tid in gold_candidates)
        result[gold] = (
            bool(_is_correct_final_answer(cached_cot, gold)),
            float(gold_lp),
        )
    return result


def _gold_answer_logprob_from_generation_scores(
    model, tokenizer, prompt_ids, continuation_ids, gold_answer, generation_scores
):
    """
    Recover the gold-answer next-token logprob from generation scores when the
    relevant decoding step was materialized during generation.

    Falls back to a dedicated forward pass only when the model stopped exactly
    at the boundary, so no post-boundary score exists in `generation_scores`.
    """
    boundary_end = _boundary_end_index(tokenizer, continuation_ids)
    if boundary_end is None:
        return np.nan

    gold_candidates = _gold_first_token_candidates(tokenizer, gold_answer)
    if boundary_end < len(generation_scores):
        step_scores = generation_scores[boundary_end]
        lp = F.log_softmax(step_scores[0], dim=-1)
        # Robust to leading-whitespace BPE merging: take max over all
        # plausible first-content-token variants of the gold answer.
        return max(lp[tid].item() for tid in gold_candidates)

    return _gold_answer_logprob_at_generated_boundary(
        model, tokenizer, prompt_ids, continuation_ids, gold_answer
    )


def patch_one_pair(model, tokenizer, src_prompt, tgt_prompt,
                   correct_ans_id, wrong_ans_id, scope, layers=None):
    """
    Cache activations from src (noop correct), patch into tgt (noop wrong).

    Returns:
        src_logprobs  : dict with 'correct' and 'wrong' logprob under src model
        tgt_logprobs  : dict with 'correct' and 'wrong' logprob under tgt model
        effects       : np.ndarray (n_layers [, n_heads]) of normalised patching
                        effects on answer logprob difference:
                        metric = logprob(scored/source answer)
                                 - logprob(contrast/target answer)
                        effect = (patched_metric - tgt_metric)
                                 / (src_metric - tgt_metric)
    """
    device = next(model.parameters()).device
    src_inputs = _encode(src_prompt, tokenizer, device)
    tgt_inputs = _encode(tgt_prompt, tokenizer, device)

    remove_all_hooks(model)
    state_dict, save_hooks = _register_last_token_saving_hooks(model, scope)

    # Single src forward pass: captures logprobs and fills state_dict simultaneously
    with torch.no_grad():
        out = model(**src_inputs)
        src_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
    src_correct = src_lp[correct_ans_id].item()
    src_wrong   = src_lp[wrong_ans_id].item()
    src_metric  = src_correct - src_wrong

    for h in save_hooks:
        try: h.remove()
        except: pass
    remove_all_hooks(model)

    with torch.no_grad():
        out = model(**tgt_inputs)
        tgt_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
    tgt_correct = tgt_lp[correct_ans_id].item()
    tgt_wrong   = tgt_lp[wrong_ans_id].item()
    tgt_metric  = tgt_correct - tgt_wrong

    num_layers = model.config.num_hidden_layers
    active_layers = layers if layers is not None else list(range(num_layers))
    n_heads = model.config.num_attention_heads if scope == "attn_head" else 1

    effects = np.full((num_layers, n_heads), np.nan)

    denom = src_metric - tgt_metric   # normalisation constant

    for layer_idx in active_layers:
        for head_idx in range(n_heads):
            remove_all_hooks(model)

            if scope == "layer":
                model.model.layers[layer_idx].register_forward_hook(
                    make_patching_hook(state_dict[f"layer_{layer_idx}"], [-1])
                )
            elif scope == "mlp":
                model.model.layers[layer_idx].mlp.register_forward_hook(
                    make_patching_hook(state_dict[f"layer_mlp_{layer_idx}"], [-1])
                )
            elif scope == "attn_output":
                model.model.layers[layer_idx].self_attn.register_forward_hook(
                    make_patching_hook(
                        state_dict[f"layer_attn_output_{layer_idx}"], [-1]
                    )
                )
            elif scope == "attn_head":
                model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                    make_patching_hook(
                        state_dict[f"layer_attn_head_{layer_idx}"], [-1],
                        head_idx, n_heads,
                    )
                )
            else:
                raise ValueError(f"Unknown scope: {scope}")

            with torch.no_grad():
                out = model(**tgt_inputs)
                patched_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
            patched_correct = patched_lp[correct_ans_id].item()
            patched_wrong = patched_lp[wrong_ans_id].item()
            patched_metric = patched_correct - patched_wrong

            raw_effect = patched_metric - tgt_metric
            effects[layer_idx, head_idx] = (
                raw_effect / denom if abs(denom) > 1e-6 else 0.0
            )

    remove_all_hooks(model)
    for h in save_hooks:
        try: h.remove()
        except: pass
    torch.cuda.empty_cache()

    return (
        {"correct": src_correct, "wrong": src_wrong, "metric": src_metric},
        {"correct": tgt_correct, "wrong": tgt_wrong, "metric": tgt_metric},
        effects,
    )


def patch_batch_direct(model, tokenizer, src_prompts, tgt_prompts,
                       correct_ans_ids, wrong_ans_ids, scope, layers=None):
    """
    Vectorised patch_one_pair for the direct experiment.
    All prompts must share the same token length (already guaranteed by
    build_aligned_direct_pairs). Processes B pairs per forward pass instead
    of 1, giving ~B× GPU utilisation with the same number of hook iterations.
    """
    device = next(model.parameters()).device
    B = len(src_prompts)
    src_inputs = _encode_batch(src_prompts, tokenizer, device)
    tgt_inputs = _encode_batch(tgt_prompts, tokenizer, device)

    remove_all_hooks(model)
    state_dict, save_hooks = _register_last_token_saving_hooks(model, scope)

    with torch.no_grad():
        out = model(**src_inputs)
        src_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)  # [B, vocab]
    src_correct = [src_lp[i, correct_ans_ids[i]].item() for i in range(B)]
    src_wrong   = [src_lp[i, wrong_ans_ids[i]].item()   for i in range(B)]
    src_metric  = [src_correct[i] - src_wrong[i] for i in range(B)]

    for h in save_hooks:
        try: h.remove()
        except: pass
    remove_all_hooks(model)

    with torch.no_grad():
        out = model(**tgt_inputs)
        tgt_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)
    tgt_correct = [tgt_lp[i, correct_ans_ids[i]].item() for i in range(B)]
    tgt_wrong   = [tgt_lp[i, wrong_ans_ids[i]].item()   for i in range(B)]
    tgt_metric  = [tgt_correct[i] - tgt_wrong[i] for i in range(B)]

    num_layers    = model.config.num_hidden_layers
    active_layers = layers if layers is not None else list(range(num_layers))
    n_heads = model.config.num_attention_heads if scope == "attn_head" else 1

    effects = np.full((B, num_layers, n_heads), np.nan)
    denom   = np.array(src_metric) - np.array(tgt_metric)  # [B]

    hook_handle = None
    for layer_idx in active_layers:
        for head_idx in range(n_heads):
            if hook_handle is not None:
                hook_handle.remove()

            if scope == "layer":
                hook_handle = model.model.layers[layer_idx].register_forward_hook(
                    make_patching_hook(state_dict[f"layer_{layer_idx}"], [-1])
                )
            elif scope == "mlp":
                hook_handle = model.model.layers[layer_idx].mlp.register_forward_hook(
                    make_patching_hook(state_dict[f"layer_mlp_{layer_idx}"], [-1])
                )
            elif scope == "attn_output":
                hook_handle = model.model.layers[layer_idx].self_attn.register_forward_hook(
                    make_patching_hook(
                        state_dict[f"layer_attn_output_{layer_idx}"], [-1]
                    )
                )
            elif scope == "attn_head":
                hook_handle = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                    make_patching_hook(
                        state_dict[f"layer_attn_head_{layer_idx}"], [-1],
                        head_idx, n_heads,
                    )
                )
            else:
                raise ValueError(f"Unknown scope: {scope}")

            with torch.no_grad():
                out = model(**tgt_inputs)
                patched_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)

            for i in range(B):
                patched_correct = patched_lp[i, correct_ans_ids[i]].item()
                patched_wrong = patched_lp[i, wrong_ans_ids[i]].item()
                patched_metric = patched_correct - patched_wrong
                raw_effect = patched_metric - tgt_metric[i]
                effects[i, layer_idx, head_idx] = (
                    raw_effect / denom[i] if abs(denom[i]) > 1e-6 else 0.0
                )

    if hook_handle is not None:
        hook_handle.remove()
    remove_all_hooks(model)

    return (
        [
            {"correct": src_correct[i], "wrong": src_wrong[i], "metric": src_metric[i]}
            for i in range(B)
        ],
        [
            {"correct": tgt_correct[i], "wrong": tgt_wrong[i], "metric": tgt_metric[i]}
            for i in range(B)
        ],
        [effects[i] for i in range(B)],
    )


def patch_one_pair_direct_span(model, tokenizer, src_prompt, tgt_prompt,
                               correct_ans_id, wrong_ans_id, scope,
                               src_indices, tgt_indices, layers=None):
    """Span-aware single-pair direct patch.

    Saves source activations at `src_indices` for every active layer, then
    re-runs the target prompt patching activations at `tgt_indices`. The score
    is still read at the final position (the answer-prediction step). Returns
    the same `(src_lp_dict, tgt_lp_dict, effects)` shape as `patch_batch_direct`
    for a single pair.
    """
    device = next(model.parameters()).device
    src_inputs = _encode(src_prompt, tokenizer, device)
    tgt_inputs = _encode(tgt_prompt, tokenizer, device)

    remove_all_hooks(model)
    state_dict, save_hooks = _register_index_saving_hooks(
        model, scope, src_indices, layers=layers,
    )
    with torch.no_grad():
        out = model(**src_inputs)
        src_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
    src_correct = src_lp[correct_ans_id].item()
    src_wrong = src_lp[wrong_ans_id].item()
    src_metric = src_correct - src_wrong
    for h in save_hooks:
        try: h.remove()
        except: pass
    remove_all_hooks(model)

    with torch.no_grad():
        out = model(**tgt_inputs)
        tgt_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
    tgt_correct = tgt_lp[correct_ans_id].item()
    tgt_wrong = tgt_lp[wrong_ans_id].item()
    tgt_metric = tgt_correct - tgt_wrong
    denom = src_metric - tgt_metric

    num_layers = model.config.num_hidden_layers
    active_layers = layers if layers is not None else list(range(num_layers))
    n_heads = model.config.num_attention_heads if scope == "attn_head" else 1
    effects = np.full((num_layers, n_heads), np.nan)

    hook_handle = None
    for layer_idx in active_layers:
        for head_idx in range(n_heads):
            if hook_handle is not None:
                hook_handle.remove()

            if scope == "layer":
                hook_handle = model.model.layers[layer_idx].register_forward_hook(
                    _make_span_patching_hook(
                        state_dict[f"layer_{layer_idx}"], tgt_indices
                    )
                )
            elif scope == "mlp":
                hook_handle = model.model.layers[layer_idx].mlp.register_forward_hook(
                    _make_span_patching_hook(
                        state_dict[f"layer_mlp_{layer_idx}"], tgt_indices
                    )
                )
            elif scope == "attn_output":
                hook_handle = model.model.layers[layer_idx].self_attn.register_forward_hook(
                    _make_span_patching_hook(
                        state_dict[f"layer_attn_output_{layer_idx}"], tgt_indices
                    )
                )
            elif scope == "attn_head":
                hook_handle = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                    _make_span_patching_hook(
                        state_dict[f"layer_attn_head_{layer_idx}"], tgt_indices,
                        head_idx, n_heads,
                    )
                )
            else:
                raise ValueError(f"Unknown scope: {scope}")

            with torch.no_grad():
                out = model(**tgt_inputs)
                patched_lp = F.log_softmax(out.logits[:, -1, :], dim=-1)[0]
            patched_correct = patched_lp[correct_ans_id].item()
            patched_wrong = patched_lp[wrong_ans_id].item()
            patched_metric = patched_correct - patched_wrong
            raw_effect = patched_metric - tgt_metric
            effects[layer_idx, head_idx] = (
                raw_effect / denom if abs(denom) > 1e-6 else 0.0
            )

    if hook_handle is not None:
        hook_handle.remove()
    remove_all_hooks(model)
    torch.cuda.empty_cache()

    return (
        {"correct": src_correct, "wrong": src_wrong, "metric": src_metric},
        {"correct": tgt_correct, "wrong": tgt_wrong, "metric": tgt_metric},
        effects,
    )


def patch_one_pair_cot_boundary(model, tokenizer, src_prompt, tgt_prompt,
                                gold_answer, scope, max_new_tokens=1024,
                                layers=None, src_question_span=None,
                                tgt_question_span=None,
                                allow_truncated_span=False,
                                cached_baseline=None,
                                src_indices_override=None,
                                tgt_indices_override=None,
                                patch_positions_tag=None,
                                target_gold_answer=None):
    """
    Patch the source question span into the target question span, then freely
    generate CoT.

    Returns dict with arrays shaped (n_layers [, n_heads]):
        correctness_effects: 1.0 when the patched run ends correct, else 0.0
        gold_logprobs: gold-answer logprob at the run's generated '####' boundary
        gold_logprob_deltas: patched gold_logprobs minus unpatched target baseline
        unpatched_target_correct: scalar bool
        unpatched_target_gold_logprob: scalar float or nan
    """
    device = next(model.parameters()).device
    src_inputs = _encode(src_prompt, tokenizer, device)
    tgt_inputs = _encode(tgt_prompt, tokenizer, device)

    if src_indices_override is not None:
        src_indices = list(src_indices_override)
        tgt_indices = list(tgt_indices_override) if tgt_indices_override is not None else []
        skipped_mismatched_span = (
            len(src_indices) == 0
            or len(src_indices) != len(tgt_indices)
        )
    elif src_question_span is None:
        src_indices = [src_inputs["input_ids"].shape[1] - 1]
        tgt_indices = [tgt_inputs["input_ids"].shape[1] - 1]
        skipped_mismatched_span = False
    else:
        src_indices, tgt_indices = _cot_patch_indices(
            src_question_span[0], src_question_span[1],
            tgt_question_span[0], tgt_question_span[1],
            allow_truncated_span=allow_truncated_span,
        )
        skipped_mismatched_span = len(src_indices) == 0

    if patch_positions_tag is None:
        if src_indices_override is not None:
            patch_positions_tag = "value_tokens"
        elif src_question_span is not None:
            patch_positions_tag = "question_span"
        else:
            patch_positions_tag = "prompt_end"

    num_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads if scope == "attn_head" else 1
    target_gold_active = (
        target_gold_answer is not None
        and str(target_gold_answer) != str(gold_answer)
    )
    if skipped_mismatched_span:
        nan_effects = np.full((num_layers, n_heads), np.nan)
        return {
            "correctness_effects": nan_effects,
            "gold_logprobs": nan_effects.copy(),
            "gold_logprob_deltas": nan_effects.copy(),
            "normalized_gold_logprob_effects": nan_effects.copy(),
            "source_gold_logprob": np.nan,
            "source_minus_target_gold_logprob": np.nan,
            "unpatched_target_correct": False,
            "unpatched_target_gold_logprob": np.nan,
            "target_gold_logprobs": nan_effects.copy(),
            "target_gold_logprob_deltas": nan_effects.copy(),
            "normalized_target_gold_logprob_effects": nan_effects.copy(),
            "target_emit_success": nan_effects.copy(),
            "patched_emitted_answers": [None] * num_layers,
            "hash_emitted": nan_effects.copy(),
            "unpatched_source_target_gold_logprob": np.nan,
            "unpatched_target_target_gold_logprob": np.nan,
            "target_gold_answer": (str(target_gold_answer)
                                   if target_gold_answer is not None else None),
            "patch_positions": patch_positions_tag,
            "num_patched_tokens": 0,
            "allow_truncated_span": bool(allow_truncated_span),
            "source_q_start": int(src_question_span[0]) if src_question_span is not None else None,
            "source_q_end": int(src_question_span[1]) if src_question_span is not None else None,
            "target_q_start": int(tgt_question_span[0]) if tgt_question_span is not None else None,
            "target_q_end": int(tgt_question_span[1]) if tgt_question_span is not None else None,
            "source_question_span_len": (
                int(src_question_span[1] - src_question_span[0] + 1)
                if src_question_span is not None else None
            ),
            "target_question_span_len": (
                int(tgt_question_span[1] - tgt_question_span[0] + 1)
                if tgt_question_span is not None else None
            ),
            "skipped_mismatched_span": True,
        }

    remove_all_hooks(model)
    state_dict, save_hooks = _register_index_saving_hooks(
        model, scope, src_indices, layers=layers
    )

    with torch.no_grad():
        model(**src_inputs)

    for h in save_hooks:
        try:
            h.remove()
        except Exception:
            pass
    remove_all_hooks(model)

    active_layers = layers if layers is not None else list(range(num_layers))
    correctness_effects = np.full((num_layers, n_heads), np.nan)
    gold_logprobs = np.full((num_layers, n_heads), np.nan)
    target_gold_logprobs = np.full((num_layers, n_heads), np.nan)
    target_emit_success = np.full((num_layers, n_heads), np.nan)
    hash_emitted = np.full((num_layers, n_heads), np.nan)
    patched_emitted_answers = [None] * num_layers
    if scope == "attn_head":
        # Per-layer × per-head; keep emit strings keyed by layer only (the
        # patched_generation differs per head but storing 80×n_heads strings
        # is excessive — keep per-layer last for diagnostic visibility).
        patched_emitted_answers_per_head = [
            [None] * n_heads for _ in range(num_layers)
        ]
    else:
        patched_emitted_answers_per_head = None

    src_prompt_ids = src_inputs["input_ids"][0].detach().cpu().tolist()
    tgt_prompt_ids = tgt_inputs["input_ids"][0].detach().cpu().tolist()

    unpatched_source_target_gold_logprob = float("nan")
    unpatched_target_target_gold_logprob = float("nan")
    unpatched_source_target_gold_correct = False
    unpatched_target_target_gold_correct = False
    if cached_baseline is not None:
        unpatched_source_correct = bool(cached_baseline["unpatched_source_correct"])
        unpatched_source_gold_logprob = float(
            cached_baseline["unpatched_source_gold_logprob"]
        )
        unpatched_target_correct = bool(cached_baseline["unpatched_target_correct"])
        unpatched_target_gold_logprob = float(
            cached_baseline["unpatched_target_gold_logprob"]
        )
        if target_gold_active:
            if "unpatched_source_target_gold_logprob" not in cached_baseline:
                raise KeyError(
                    "Baseline cache is missing target-gold fields required for "
                    "full-metrics cot_boundary patching. Re-run baselines "
                    "(`--baselines_only`) with the updated code so target-gold "
                    "log P is computed on both unpatched runs."
                )
            unpatched_source_target_gold_logprob = float(
                cached_baseline["unpatched_source_target_gold_logprob"]
            )
            unpatched_target_target_gold_logprob = float(
                cached_baseline["unpatched_target_target_gold_logprob"]
            )
            unpatched_source_target_gold_correct = bool(
                cached_baseline.get("unpatched_source_target_gold_correct", False)
            )
            unpatched_target_target_gold_correct = bool(
                cached_baseline.get("unpatched_target_target_gold_correct", False)
            )
    else:
        with torch.no_grad():
            unpatched_source_ids, unpatched_source_generation, unpatched_source_scores = _greedy_generate(
                model, tokenizer, src_inputs, max_new_tokens=max_new_tokens,
                return_scores=True,
            )
        unpatched_source_correct = _is_correct_final_answer(
            unpatched_source_generation, gold_answer
        )
        unpatched_source_gold_logprob = _gold_answer_logprob_from_generation_scores(
            model, tokenizer, src_prompt_ids, unpatched_source_ids, gold_answer,
            unpatched_source_scores,
        )
        if target_gold_active:
            unpatched_source_target_gold_correct = _is_correct_final_answer(
                unpatched_source_generation, target_gold_answer
            )
            unpatched_source_target_gold_logprob = _gold_answer_logprob_from_generation_scores(
                model, tokenizer, src_prompt_ids, unpatched_source_ids,
                target_gold_answer, unpatched_source_scores,
            )

        with torch.no_grad():
            unpatched_ids, unpatched_generation, unpatched_scores = _greedy_generate(
                model, tokenizer, tgt_inputs, max_new_tokens=max_new_tokens,
                return_scores=True,
            )
        unpatched_target_correct = _is_correct_final_answer(unpatched_generation, gold_answer)
        unpatched_target_gold_logprob = _gold_answer_logprob_from_generation_scores(
            model, tokenizer, tgt_prompt_ids, unpatched_ids, gold_answer,
            unpatched_scores,
        )
        if target_gold_active:
            unpatched_target_target_gold_correct = _is_correct_final_answer(
                unpatched_generation, target_gold_answer
            )
            unpatched_target_target_gold_logprob = _gold_answer_logprob_from_generation_scores(
                model, tokenizer, tgt_prompt_ids, unpatched_ids, target_gold_answer,
                unpatched_scores,
            )

    hook_handle = None
    if not skipped_mismatched_span:
        for layer_idx in active_layers:
            for head_idx in range(n_heads):
                if hook_handle is not None:
                    hook_handle.remove()

                if scope == "layer":
                    hook_handle = model.model.layers[layer_idx].register_forward_hook(
                        _make_span_patching_hook(
                            state_dict[f"layer_{layer_idx}"], tgt_indices
                        )
                    )
                elif scope == "mlp":
                    hook_handle = model.model.layers[layer_idx].mlp.register_forward_hook(
                        _make_span_patching_hook(
                            state_dict[f"layer_mlp_{layer_idx}"], tgt_indices
                        )
                    )
                elif scope == "attn_output":
                    hook_handle = model.model.layers[layer_idx].self_attn.register_forward_hook(
                        _make_span_patching_hook(
                            state_dict[f"layer_attn_output_{layer_idx}"], tgt_indices
                        )
                    )
                elif scope == "attn_head":
                    hook_handle = model.model.layers[layer_idx].self_attn.o_proj.register_forward_hook(
                        _make_span_patching_hook(
                            state_dict[f"layer_attn_head_{layer_idx}"], tgt_indices,
                            head_idx, n_heads,
                        )
                    )
                else:
                    raise ValueError(f"Unknown scope: {scope}")

                with torch.no_grad():
                    patched_ids, patched_generation, patched_scores = _greedy_generate(
                        model, tokenizer, tgt_inputs, max_new_tokens=max_new_tokens,
                        return_scores=True,
                    )

                correctness_effects[layer_idx, head_idx] = float(
                    _is_correct_final_answer(patched_generation, gold_answer)
                )
                gold_logprobs[layer_idx, head_idx] = (
                    _gold_answer_logprob_from_generation_scores(
                        model, tokenizer, tgt_prompt_ids, patched_ids, gold_answer,
                        patched_scores,
                    )
                )
                if target_gold_active:
                    target_emit_success[layer_idx, head_idx] = float(
                        _is_correct_final_answer(patched_generation, target_gold_answer)
                    )
                    target_gold_logprobs[layer_idx, head_idx] = (
                        _gold_answer_logprob_from_generation_scores(
                            model, tokenizer, tgt_prompt_ids, patched_ids,
                            target_gold_answer, patched_scores,
                        )
                    )
                # "####" emission sanity check: did the patched generation
                # actually reach an answer boundary (vs. ramble until cutoff)?
                gen_text = str(patched_generation)
                hash_emitted[layer_idx, head_idx] = float("####" in gen_text)
                emitted = _extract_cot_answer_text(gen_text)
                if patched_emitted_answers_per_head is not None:
                    patched_emitted_answers_per_head[layer_idx][head_idx] = emitted
                # For non-attn_head scopes (n_heads=1) keep the per-layer entry
                # so result rows stay flat; for attn_head, store the head=0 slot
                # as the per-layer summary and keep the full per-head list too.
                if patched_emitted_answers[layer_idx] is None:
                    patched_emitted_answers[layer_idx] = emitted

    if hook_handle is not None:
        hook_handle.remove()
    remove_all_hooks(model)
    for h in save_hooks:
        try:
            h.remove()
        except Exception:
            pass
    torch.cuda.empty_cache()
    source_minus_target_gold_logprob = (
        unpatched_source_gold_logprob - unpatched_target_gold_logprob
    )
    if abs(source_minus_target_gold_logprob) > 1e-6:
        normalized_gold_logprob_effects = (
            (gold_logprobs - unpatched_target_gold_logprob)
            / source_minus_target_gold_logprob
        )
    else:
        normalized_gold_logprob_effects = np.full_like(gold_logprobs, np.nan)

    # Target-gold normalization: 1 = target's natural reading (its own unpatched
    # gold_logprob), 0 = source's reading (target_gold under source's unpatched
    # run). For low-effect patches this should stay near 1; for high-effect
    # patches it falls toward 0 (target's answer is suppressed).
    if target_gold_active:
        target_minus_source_target_gold_logprob = (
            unpatched_target_target_gold_logprob - unpatched_source_target_gold_logprob
        )
        if abs(target_minus_source_target_gold_logprob) > 1e-6:
            normalized_target_gold_logprob_effects = (
                (target_gold_logprobs - unpatched_source_target_gold_logprob)
                / target_minus_source_target_gold_logprob
            )
        else:
            normalized_target_gold_logprob_effects = np.full_like(
                target_gold_logprobs, np.nan
            )
        target_gold_logprob_deltas = (
            target_gold_logprobs - unpatched_target_target_gold_logprob
        )
    else:
        normalized_target_gold_logprob_effects = np.full_like(
            target_gold_logprobs, np.nan
        )
        target_gold_logprob_deltas = np.full_like(target_gold_logprobs, np.nan)
        target_minus_source_target_gold_logprob = float("nan")

    return {
        "correctness_effects": correctness_effects,
        "gold_logprobs": gold_logprobs,
        "gold_logprob_deltas": gold_logprobs - unpatched_target_gold_logprob,
        "normalized_gold_logprob_effects": normalized_gold_logprob_effects,
        "target_gold_logprobs": target_gold_logprobs,
        "target_gold_logprob_deltas": target_gold_logprob_deltas,
        "normalized_target_gold_logprob_effects": normalized_target_gold_logprob_effects,
        "target_emit_success": target_emit_success,
        "patched_emitted_answers": patched_emitted_answers,
        "patched_emitted_answers_per_head": patched_emitted_answers_per_head,
        "hash_emitted": hash_emitted,
        "target_gold_answer": (str(target_gold_answer)
                               if target_gold_answer is not None else None),
        "unpatched_source_correct": bool(unpatched_source_correct),
        "unpatched_source_gold_logprob": unpatched_source_gold_logprob,
        "unpatched_source_target_gold_correct": bool(unpatched_source_target_gold_correct),
        "unpatched_source_target_gold_logprob": unpatched_source_target_gold_logprob,
        "source_minus_target_gold_logprob": source_minus_target_gold_logprob,
        "unpatched_target_correct": bool(unpatched_target_correct),
        "unpatched_target_gold_logprob": unpatched_target_gold_logprob,
        "unpatched_target_target_gold_correct": bool(unpatched_target_target_gold_correct),
        "unpatched_target_target_gold_logprob": unpatched_target_target_gold_logprob,
        "target_minus_source_target_gold_logprob": target_minus_source_target_gold_logprob,
        "patch_positions": patch_positions_tag,
        "num_patched_tokens": int(len(src_indices)),
        "src_indices": [int(i) for i in src_indices],
        "tgt_indices": [int(i) for i in tgt_indices],
        "allow_truncated_span": bool(allow_truncated_span),
        "source_q_start": int(src_question_span[0]) if src_question_span is not None else None,
        "source_q_end": int(src_question_span[1]) if src_question_span is not None else None,
        "target_q_start": int(tgt_question_span[0]) if tgt_question_span is not None else None,
        "target_q_end": int(tgt_question_span[1]) if tgt_question_span is not None else None,
        "source_question_span_len": (
            int(src_question_span[1] - src_question_span[0] + 1)
            if src_question_span is not None else None
        ),
        "target_question_span_len": (
            int(tgt_question_span[1] - tgt_question_span[0] + 1)
            if tgt_question_span is not None else None
        ),
        "skipped_mismatched_span": bool(skipped_mismatched_span),
    }


def _first_ans_token(tokenizer, answer_str):
    """Return the first token id of the answer string."""
    return tokenizer.encode(str(answer_str).strip(), add_special_tokens=False)[0]


def _gold_first_token_candidates(tokenizer, answer_str):
    """Return all plausible token ids the model could emit as the first
    content token of `answer_str`, accounting for BPE merging of leading
    whitespace into the number token.

    Llama-3 BPE often merges a leading space into the number (so the model
    emits a single token like `_26` instead of `[" ", "26"]`); the same can
    happen for tabs or newlines. Without this, log P of the bare encoding
    misses the merged variant the model actually emits, deflating the
    measurement. By enumerating prefixes and collecting the first content
    token from each, we cover the variants the model might pick. Downstream
    `max` over their logprobs recovers the actually-emitted variant.
    """
    gold = str(answer_str).strip()
    candidates = set()
    for prefix in ("", " ", "\t", "\n"):
        tokens = tokenizer.encode(prefix + gold, add_special_tokens=False)
        for tid in tokens:
            piece = tokenizer.decode([tid], skip_special_tokens=False)
            if piece.strip():  # skip whitespace-only tokens; take first content
                candidates.add(int(tid))
                break
    return sorted(candidates)


def _direct_answer_columns(pairs, canonical_direction, pair_dataset):
    """
    Return (scored_answer_col, contrast_answer_col).

    The legacy noop/filler experiments always score the c-side answer, using
    the reverse direction as a corruption test. For P1 vs padded GSM-Sym, both
    sides are valid CoT-correct questions with different answers, so each
    direction scores the source-side answer in the target prompt.
    """
    c_col = "patch_answer_c" if "patch_answer_c" in pairs.columns else "original_direct_answer_c"
    w_col = "patch_answer_w" if "patch_answer_w" in pairs.columns else "original_direct_answer_w"

    if (
        _is_p1_padded_dataset(pair_dataset)
        or _is_gsm_sym_within_template_dataset(pair_dataset)
        or _is_noop_clean_self_ww_dataset(pair_dataset)
    ):
        if canonical_direction == "wrong_to_correct":
            return w_col, c_col
        return c_col, w_col

    return c_col, w_col


def _load_done_keys(out_path):
    """Read previously-written JSONL results and return the set of completed
    pair keys. Drops unparseable lines (torn writes from SIGTERM/SIGKILL,
    partial flushes, disk hiccups) and heals the file by rewriting it with
    only the valid records, so resume can proceed without `--overwrite`.
    Dropped pairs will simply be recomputed — never silently discards a
    parseable record. Duplicate keys (pathological; shouldn't occur in normal
    operation) are deduped in memory but the file is left untouched."""
    if not out_path.exists() or out_path.stat().st_size == 0:
        return set()
    with open(out_path, "rb") as f:
        raw = f.read()
    lines = raw.splitlines(keepends=True)
    done = set()
    good_lines = []
    bad_count = 0
    first_bad = None
    duplicate_count = 0
    for i, line in enumerate(lines):
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            # Blank line: not progress, but also not corruption. Preserve it
            # so we don't trigger a needless rewrite.
            good_lines.append(line)
            continue
        try:
            r = json.loads(text)
            key = (r["original_id"], r["instance_c"], r["instance_w"])
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
            bad_count += 1
            if first_bad is None:
                first_bad = (i + 1, str(e))
            # Drop this line — do NOT append to good_lines.
            continue
        if key in done:
            duplicate_count += 1
            # Keep the line in good_lines so we don't rewrite-and-drop it;
            # only the in-memory `done` set is deduped.
        else:
            done.add(key)
        good_lines.append(line if line.endswith(b"\n") else line + b"\n")

    if duplicate_count:
        print(
            f"  warning: {out_path.name} has {duplicate_count} duplicate "
            f"record(s); deduping in memory, file left as-is"
        )

    if bad_count:
        idx, reason = first_bad
        print(
            f"  warning: dropping {bad_count} corrupt line(s) from "
            f"{out_path.name} (first bad: line {idx}, {reason}); healing file"
        )
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp_path, "wb") as f:
            f.writelines(good_lines)
        os.replace(tmp_path, out_path)
    return done


def run_noop_patching_direct(model, tokenizer, model_id, model_name, scope,
                             max_pairs_per_orig=None, layers=None, batch_size=1,
                             direction="correct_to_wrong", pair_dataset="noop_clean_self",
                             overwrite=False, patch_positions="prompt_end",
                             allow_truncated_span=False):
    if patch_positions not in ("prompt_end", "question_span"):
        raise ValueError(f"Unknown patch_positions: {patch_positions}")
    span_mode = (patch_positions == "question_span")
    result_dir = _result_dir(pair_dataset)
    result_dir.mkdir(parents=True, exist_ok=True)
    canonical_direction = _canonical_direction(direction)

    pairs = build_aligned_direct_pairs(
        tokenizer, model_id, max_pairs_per_orig=max_pairs_per_orig,
        pair_dataset=pair_dataset, experiment="direct",
        compute_question_spans=span_mode,
    )
    if pairs.empty:
        print("No token-aligned direct pairs found after filtering.")
        return

    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    dir_tag = _direction_tag(direction, pair_dataset)
    span_tag = (
        "_question_span" + ("_truncated" if allow_truncated_span else "")
        if span_mode else ""
    )
    out_path = _result_path(
        model_name, "direct", scope, span_tag, layers_tag, dir_tag, pair_dataset,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if not overwrite:
        done = _load_done_keys(out_path)
        if done:
            print(f"Resuming — {len(done)} pairs already done")

    model.eval()

    # Pre-compute answer token ids and drop degenerate pairs up front.
    pairs = pairs.copy()
    scored_answer_col, contrast_answer_col = _direct_answer_columns(
        pairs, canonical_direction, pair_dataset
    )
    pairs["scored_ans_id"] = pairs[scored_answer_col].apply(
        lambda a: _first_ans_token(tokenizer, a)
    )
    pairs["contrast_ans_id"] = pairs[contrast_answer_col].apply(
        lambda a: _first_ans_token(tokenizer, a)
    )
    n_before = len(pairs)
    pairs = pairs[pairs["scored_ans_id"] != pairs["contrast_ans_id"]].copy()
    _log_filter(
        "run_noop_patching_direct",
        n_before,
        len(pairs),
        "same first answer token",
    )

    with open(out_path, "w" if overwrite else "a") as f_out:
        pbar = tqdm(total=len(pairs), desc=scope, unit="pair")
        skipped_done = 0
        written_pairs = 0

        # Group by source/target prompt lengths so every batch needs no padding.
        group_cols = ["tok_len_c", "tok_len_w"] if {"tok_len_c", "tok_len_w"}.issubset(pairs.columns) else ["tok_len"]
        for _, group in pairs.groupby(group_cols):
            undone = group[~group.apply(
                lambda r: (int(r["original_id"]), int(r["instance_c"]), int(r["instance_w"])) in done,
                axis=1,
            )].reset_index(drop=True)

            skipped_done += len(group) - len(undone)
            pbar.update(len(group) - len(undone))  # credit already-done pairs

            effective_bs = 1 if span_mode else batch_size
            for i in range(0, len(undone), effective_bs):
                batch = undone.iloc[i : i + effective_bs]

                if canonical_direction == "wrong_to_correct":
                    src_prompts_batch = batch["full_prompt_w"].tolist()
                    tgt_prompts_batch = batch["full_prompt_c"].tolist()
                    src_q_start_col, src_q_end_col = "q_start_w", "q_end_w"
                    tgt_q_start_col, tgt_q_end_col = "q_start_c", "q_end_c"
                else:
                    src_prompts_batch = batch["full_prompt_c"].tolist()
                    tgt_prompts_batch = batch["full_prompt_w"].tolist()
                    src_q_start_col, src_q_end_col = "q_start_c", "q_end_c"
                    tgt_q_start_col, tgt_q_end_col = "q_start_w", "q_end_w"

                if not span_mode:
                    src_lps, tgt_lps, effects_list = patch_batch_direct(
                        model, tokenizer,
                        src_prompts_batch,
                        tgt_prompts_batch,
                        batch["scored_ans_id"].tolist(),
                        batch["contrast_ans_id"].tolist(),
                        scope=scope, layers=layers,
                    )
                    span_meta_list = [None] * len(batch)
                else:
                    src_lps, tgt_lps, effects_list, span_meta_list = [], [], [], []
                    for (_, row) in batch.iterrows():
                        src_qs, src_qe = int(row[src_q_start_col]), int(row[src_q_end_col])
                        tgt_qs, tgt_qe = int(row[tgt_q_start_col]), int(row[tgt_q_end_col])
                        src_indices, tgt_indices = _cot_patch_indices(
                            src_qs, src_qe, tgt_qs, tgt_qe,
                            allow_truncated_span=allow_truncated_span,
                        )
                        meta = {
                            "patch_positions": "question_span",
                            "source_q_start": src_qs,
                            "source_q_end": src_qe,
                            "target_q_start": tgt_qs,
                            "target_q_end": tgt_qe,
                            "source_question_span_len": src_qe - src_qs + 1,
                            "target_question_span_len": tgt_qe - tgt_qs + 1,
                            "num_patched_tokens": len(src_indices),
                            "allow_truncated_span": bool(allow_truncated_span),
                            "skipped_mismatched_span": len(src_indices) == 0,
                        }
                        if not src_indices:
                            nan_eff = np.full(
                                (model.config.num_hidden_layers,
                                 model.config.num_attention_heads if scope == "attn_head" else 1),
                                np.nan,
                            )
                            src_lps.append({"correct": float("nan"), "wrong": float("nan"), "metric": float("nan")})
                            tgt_lps.append({"correct": float("nan"), "wrong": float("nan"), "metric": float("nan")})
                            effects_list.append(nan_eff)
                        else:
                            src_lp, tgt_lp, eff = patch_one_pair_direct_span(
                                model, tokenizer,
                                row["full_prompt_w"] if canonical_direction == "wrong_to_correct" else row["full_prompt_c"],
                                row["full_prompt_c"] if canonical_direction == "wrong_to_correct" else row["full_prompt_w"],
                                int(row["scored_ans_id"]),
                                int(row["contrast_ans_id"]),
                                scope=scope,
                                src_indices=src_indices,
                                tgt_indices=tgt_indices,
                                layers=layers,
                            )
                            src_lps.append(src_lp)
                            tgt_lps.append(tgt_lp)
                            effects_list.append(eff)
                        span_meta_list.append(meta)

                for j, (_, row) in enumerate(batch.iterrows()):
                    result = {
                        "original_id": int(row["original_id"]),
                        "instance_c":  int(row["instance_c"]),
                        "instance_w":  int(row["instance_w"]),
                        "tok_len":     int(row["tok_len"]),
                        "tok_len_c":   int(row["tok_len_c"]) if "tok_len_c" in row else int(row["tok_len"]),
                        "tok_len_w":   int(row["tok_len_w"]) if "tok_len_w" in row else int(row["tok_len"]),
                        "direction":   direction,
                        "scored_ans":  str(row[scored_answer_col]),
                        "contrast_ans": str(row[contrast_answer_col]),
                        "correct_ans": str(row[scored_answer_col]),
                        "wrong_ans":   str(row[contrast_answer_col]),
                        "src_lp":      src_lps[j],
                        "tgt_lp":      tgt_lps[j],
                        "src_metric":  src_lps[j]["metric"],
                        "tgt_metric":  tgt_lps[j]["metric"],
                        "metric_name": "answer_logprob_diff",
                        "effects":     effects_list[j].tolist(),
                    }
                    if span_meta_list[j] is not None:
                        result.update(span_meta_list[j])
                    f_out.write(json.dumps(result) + "\n")
                    written_pairs += 1
                f_out.flush()
                pbar.update(len(batch))

        pbar.close()
    print(
        "run_noop_patching_direct: "
        f"total_pairs_after_filters={len(pairs)}, skipped_already_done={skipped_done}, "
        f"written_pairs={written_pairs}"
    )


def run_noop_patching_cot_boundary(model, tokenizer, model_id, model_name, scope,
                                   max_pairs_per_orig=None, layers=None,
                                   max_new_tokens=1024, direction="correct_to_wrong",
                                   pair_dataset="noop_clean_self", overwrite=False,
                                   allow_truncated_span=False,
                                   allow_no_baseline_cache=False,
                                   patch_positions="question_span",
                                   value_tokens_position_exact=False,
                                   max_pairs_per_orig_post_filter=None):
    if patch_positions not in ("question_span", "value_tokens"):
        raise ValueError(
            "cot_boundary patch_positions must be 'question_span' or "
            f"'value_tokens', got: {patch_positions!r}"
        )
    if patch_positions == "value_tokens" and not _is_gsm_sym_within_template_dataset(pair_dataset):
        raise ValueError(
            "patch_positions=value_tokens requires pair_dataset="
            "gsm_sym_within_template (needs symbol_binding fields)."
        )
    if value_tokens_position_exact and patch_positions != "value_tokens":
        raise ValueError(
            "--value_tokens_position_exact only applies to "
            "--patch_positions value_tokens."
        )

    result_dir = _result_dir(pair_dataset)
    result_dir.mkdir(parents=True, exist_ok=True)
    canonical_direction = _canonical_direction(direction)

    pairs = build_cot_boundary_pairs(
        tokenizer, model_id, max_pairs_per_orig=max_pairs_per_orig,
        pair_dataset=pair_dataset, experiment="cot_boundary",
    )
    if pairs.empty:
        print("No CoT boundary pairs found after filtering.")
        return

    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    if patch_positions == "value_tokens":
        # Truncation flag is meaningless for non-contiguous operand patches;
        # operand alignment uses the per-pair token-count filter instead.
        # Strict position-exact runs go to a sibling directory so they
        # don't clobber the looser filter's results.
        span_tag = (
            "_value_tokens_position_exact" if value_tokens_position_exact
            else "_value_tokens"
        )
    else:
        span_tag = "_question_span" + ("_truncated" if allow_truncated_span else "")
    dir_tag = _direction_tag(direction, pair_dataset)
    out_path = _result_path(
        model_name, "cot_boundary", scope, span_tag, layers_tag, dir_tag,
        pair_dataset,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Baselines depend only on the unpatched source/target prompts and gold
    # answers — independent of which token subset gets patched. Share the
    # `question_span` baselines so value_tokens runs don't re-compute them.
    baselines_span_tag = "_question_span" + (
        "_truncated" if allow_truncated_span else ""
    )
    baselines_path = _baselines_path(
        model_name, "cot_boundary", pair_dataset, dir_tag, span_tag=baselines_span_tag,
    )
    baseline_cache = _load_cot_boundary_baselines(baselines_path)
    if baseline_cache:
        print(
            f"Loaded {len(baseline_cache)} cached unpatched baselines from "
            f"{baselines_path}"
        )
    elif allow_no_baseline_cache:
        print(
            f"No baseline cache at {baselines_path}; "
            "unpatched generations will be computed inline (override via "
            "--allow_no_baseline_cache). Each layer-shard will regenerate "
            "the unpatched source/target CoT per pair (~3x slower than the "
            "cached path)."
        )
    else:
        raise SystemExit(
            "FATAL: cot_boundary patching requires a precomputed baseline "
            "cache, but none was found.\n"
            f"  Expected at: {baselines_path}\n"
            f"  (or sibling .shardIofN.jsonl shard files)\n\n"
            "Each layer-shard otherwise regenerates the unpatched source and "
            "target CoTs autoregressively — running 80 layer-shards without a "
            "cache repeats ~2 × N_pairs generations 80 times.\n\n"
            "Fix one of these:\n"
            "  (1) Run baselines first:\n"
            "        python run_noop_activation_patching.py \\\n"
            f"          --experiment cot_boundary --pair_dataset {pair_dataset} \\\n"
            f"          --direction {direction} --baselines_only\n"
            "      (parallelize via --array=0-N --num_baseline_shards N).\n"
            "  (2) Or explicitly accept the cost: pass --allow_no_baseline_cache."
        )

    done = set()
    if not overwrite:
        done = _load_done_keys(out_path)
        if done:
            print(f"Resuming — {len(done)} pairs already done")

    model.eval()

    skipped_mismatched_span = 0
    skipped_done = 0
    attempted_pairs = 0
    written_pairs = 0
    skipped_value_token_mismatch = 0
    skipped_value_token_position_mismatch = 0
    skipped_post_filter_quota = 0
    # Per-template counter for the post-filter per-template cap. Only used
    # in value_tokens mode where `max_pairs_per_orig_post_filter` is set;
    # pairs are counted *after* the alignment + position-exact filters pass
    # (and after resume-skip), so the cap guarantees ≤N actually-patched
    # pairs per template, not ≤N candidates considered.
    post_filter_per_template = (
        {} if (patch_positions == "value_tokens"
               and max_pairs_per_orig_post_filter is not None) else None
    )
    with open(out_path, "w" if overwrite else "a") as f_out:
        for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc=scope):
            key = (
                int(row["original_id"]),
                int(row["instance_c"]),
                int(row["instance_w"]),
            )
            if key in done:
                skipped_done += 1
                continue
            attempted_pairs += 1

            if canonical_direction == "wrong_to_correct":
                src_prompt, src_q_start, src_q_end = _apply_cot_template_with_question_span(
                    row["question_w"], tokenizer
                )
                tgt_prompt, tgt_q_start, tgt_q_end = _apply_cot_template_with_question_span(
                    row["question_c"], tokenizer
                )
                gold_answer = (
                    str(row["answer_w"])
                    if _uses_source_answer_as_gold(pair_dataset)
                    else str(row["answer_c"])
                )
                source_cot_answer = str(row["original_cot_answer_w"])
                target_cot_answer = str(row["original_cot_answer_c"])
            else:
                src_prompt, src_q_start, src_q_end = _apply_cot_template_with_question_span(
                    row["question_c"], tokenizer
                )
                tgt_prompt, tgt_q_start, tgt_q_end = _apply_cot_template_with_question_span(
                    row["question_w"], tokenizer
                )
                gold_answer = (
                    str(row["answer_c"])
                    if _uses_source_answer_as_gold(pair_dataset)
                    else str(row["answer_w"])
                )
                source_cot_answer = str(row["original_cot_answer_c"])
                target_cot_answer = str(row["original_cot_answer_w"])
            # For datasets that discover gold at baseline time (e.g.
            # noop_clean_self_ww), prefer the cached gold over the pair-row
            # gold so the layer sweep tracks the same answer the baseline pass
            # actually scored against. Skip pairs that the baseline pass
            # dropped (degenerate same-answer regen, no source answer, etc.).
            cached = baseline_cache.get(key)
            if pair_dataset == "noop_clean_self_ww":
                if cached is None:
                    skipped_done += 1  # treat as already-handled-(skipped)-by-baseline
                    continue
                gold_answer = str(cached["gold_answer"])
            elif cached is not None and "gold_answer" in cached:
                # Other datasets: cache and row should agree, but trust cache
                # if present to keep gold consistent end-to-end.
                gold_answer = str(cached["gold_answer"])

            # Target-side gold for full-metric recording. None for pair
            # datasets where source and target share the same correct answer.
            target_gold_answer = _pair_target_gold(
                row, canonical_direction, pair_dataset
            )

            value_token_meta = None
            if patch_positions == "value_tokens":
                if canonical_direction == "wrong_to_correct":
                    src_question = row["question_w"]
                    tgt_question = row["question_c"]
                    src_symbolic_question = str(row.get("symbolic_question_w", ""))
                    tgt_symbolic_question = str(row.get("symbolic_question_c", ""))
                    src_symbol_binding = str(row.get("symbol_binding_w", ""))
                    tgt_symbol_binding = str(row.get("symbol_binding_c", ""))
                else:
                    src_question = row["question_c"]
                    tgt_question = row["question_w"]
                    src_symbolic_question = str(row.get("symbolic_question_c", ""))
                    tgt_symbolic_question = str(row.get("symbolic_question_w", ""))
                    src_symbol_binding = str(row.get("symbol_binding_c", ""))
                    tgt_symbol_binding = str(row.get("symbol_binding_w", ""))
                src_pos = _value_token_positions(
                    src_prompt, src_question, src_symbol_binding, tokenizer,
                    symbolic_question=src_symbolic_question,
                )
                tgt_pos = _value_token_positions(
                    tgt_prompt, tgt_question, tgt_symbol_binding, tokenizer,
                    symbolic_question=tgt_symbolic_question,
                )
                src_indices, tgt_indices = _aligned_value_token_indices(src_pos, tgt_pos)
                if src_indices is None:
                    skipped_value_token_mismatch += 1
                    continue
                if value_tokens_position_exact:
                    # Strict filter: drop pairs unless source/target prompts
                    # have identical token counts AND every operand sits at
                    # exactly the same absolute positions on both sides.
                    # This eliminates any positional-encoding confound and
                    # makes value_tokens patching a pure content intervention.
                    s_total = len(tokenizer(src_prompt, add_special_tokens=False)["input_ids"])
                    t_total = len(tokenizer(tgt_prompt, add_special_tokens=False)["input_ids"])
                    positions_align = all(
                        src_pos.get(k) == tgt_pos.get(k)
                        for k in set(src_pos) | set(tgt_pos)
                    )
                    if s_total != t_total or not positions_align:
                        skipped_value_token_position_mismatch += 1
                        continue
                if post_filter_per_template is not None:
                    oid = int(row["original_id"])
                    if post_filter_per_template.get(oid, 0) >= max_pairs_per_orig_post_filter:
                        skipped_post_filter_quota += 1
                        continue
                    post_filter_per_template[oid] = (
                        post_filter_per_template.get(oid, 0) + 1
                    )
                value_token_meta = {
                    "src_value_positions": {k: list(map(int, v))
                                            for k, v in src_pos.items()},
                    "tgt_value_positions": {k: list(map(int, v))
                                            for k, v in tgt_pos.items()},
                }
                metrics = patch_one_pair_cot_boundary(
                    model,
                    tokenizer,
                    src_prompt,
                    tgt_prompt,
                    gold_answer=gold_answer,
                    scope=scope,
                    max_new_tokens=max_new_tokens,
                    layers=layers,
                    src_question_span=(src_q_start, src_q_end),
                    tgt_question_span=(tgt_q_start, tgt_q_end),
                    allow_truncated_span=allow_truncated_span,
                    cached_baseline=cached,
                    src_indices_override=src_indices,
                    tgt_indices_override=tgt_indices,
                    patch_positions_tag="value_tokens",
                    target_gold_answer=target_gold_answer,
                )
            else:
                src_indices, tgt_indices = _cot_patch_indices(
                    src_q_start, src_q_end, tgt_q_start, tgt_q_end,
                    allow_truncated_span=allow_truncated_span,
                )
                if not src_indices:
                    skipped_mismatched_span += 1
                    continue
                metrics = patch_one_pair_cot_boundary(
                    model,
                    tokenizer,
                    src_prompt,
                    tgt_prompt,
                    gold_answer=gold_answer,
                    scope=scope,
                    max_new_tokens=max_new_tokens,
                    layers=layers,
                    src_question_span=(src_q_start, src_q_end),
                    tgt_question_span=(tgt_q_start, tgt_q_end),
                    allow_truncated_span=allow_truncated_span,
                    cached_baseline=cached,
                    target_gold_answer=target_gold_answer,
                )
            if metrics["skipped_mismatched_span"]:
                skipped_mismatched_span += 1
                continue

            result = {
                "original_id": int(row["original_id"]),
                "instance_c": int(row["instance_c"]),
                "instance_w": int(row["instance_w"]),
                "gold_answer": gold_answer,
                "gold_answer_role": (
                    "source_answer"
                    if _uses_source_answer_as_gold(pair_dataset)
                    else "target_correct_answer"
                ),
                "source_cot_answer": source_cot_answer,
                "target_cot_answer": target_cot_answer,
                "readout_name": "log P(source answer at generated #### boundary)",
                "effects": metrics["correctness_effects"].tolist(),
                "patched_final_answer_success": metrics["correctness_effects"].tolist(),
                "gold_logprobs": metrics["gold_logprobs"].tolist(),
                "patched_readout_values": metrics["gold_logprobs"].tolist(),
                "gold_logprob_deltas": metrics["gold_logprob_deltas"].tolist(),
                "readout_deltas": metrics["gold_logprob_deltas"].tolist(),
                "normalized_gold_logprob_effects": metrics["normalized_gold_logprob_effects"].tolist(),
                "normalized_effects": metrics["normalized_gold_logprob_effects"].tolist(),
                "unpatched_source_correct": metrics["unpatched_source_correct"],
                "unpatched_source_gold_logprob": metrics["unpatched_source_gold_logprob"],
                "source_baseline_readout": metrics["unpatched_source_gold_logprob"],
                "source_minus_target_gold_logprob": metrics["source_minus_target_gold_logprob"],
                "source_target_gap": metrics["source_minus_target_gold_logprob"],
                "unpatched_target_correct": metrics["unpatched_target_correct"],
                "unpatched_target_gold_logprob": metrics["unpatched_target_gold_logprob"],
                "target_baseline_readout": metrics["unpatched_target_gold_logprob"],
                "patch_positions": metrics["patch_positions"],
                "num_patched_tokens": metrics["num_patched_tokens"],
                "allow_truncated_span": metrics["allow_truncated_span"],
                "source_q_start": metrics["source_q_start"],
                "source_q_end": metrics["source_q_end"],
                "target_q_start": metrics["target_q_start"],
                "target_q_end": metrics["target_q_end"],
                "source_question_span_len": metrics["source_question_span_len"],
                "target_question_span_len": metrics["target_question_span_len"],
                "skipped_mismatched_span": metrics["skipped_mismatched_span"],
            }
            if "src_indices" in metrics:
                result["src_indices"] = metrics["src_indices"]
                result["tgt_indices"] = metrics["tgt_indices"]
            if value_token_meta is not None:
                result.update(value_token_meta)
            # Always-on extra diagnostics: emit strings + hash-completion flag.
            result["patched_emitted_answers"] = metrics["patched_emitted_answers"]
            result["hash_emitted"] = metrics["hash_emitted"].tolist()
            if metrics.get("patched_emitted_answers_per_head") is not None:
                result["patched_emitted_answers_per_head"] = (
                    metrics["patched_emitted_answers_per_head"]
                )
            # Target-side metrics: populated when target_gold_answer was given
            # (currently within_template; nan/None elsewhere so downstream
            # tooling can detect "not measured").
            if metrics.get("target_gold_answer") is not None:
                result.update({
                    "target_gold_answer": metrics["target_gold_answer"],
                    "target_gold_logprobs": metrics["target_gold_logprobs"].tolist(),
                    "target_gold_logprob_deltas": metrics["target_gold_logprob_deltas"].tolist(),
                    "normalized_target_gold_logprob_effects": metrics[
                        "normalized_target_gold_logprob_effects"
                    ].tolist(),
                    "target_emit_success": metrics["target_emit_success"].tolist(),
                    "unpatched_source_target_gold_correct":
                        metrics["unpatched_source_target_gold_correct"],
                    "unpatched_source_target_gold_logprob":
                        metrics["unpatched_source_target_gold_logprob"],
                    "unpatched_target_target_gold_correct":
                        metrics["unpatched_target_target_gold_correct"],
                    "unpatched_target_target_gold_logprob":
                        metrics["unpatched_target_target_gold_logprob"],
                    "target_minus_source_target_gold_logprob":
                        metrics["target_minus_source_target_gold_logprob"],
                })
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()
            written_pairs += 1

    print(
        "run_noop_patching_cot_boundary: "
        f"total_pairs={len(pairs)}, skipped_already_done={skipped_done}, "
        f"attempted_pairs={attempted_pairs}, written_pairs={written_pairs}, "
        f"discarded_mismatched_question_span={skipped_mismatched_span}, "
        f"discarded_value_token_mismatch={skipped_value_token_mismatch}, "
        f"discarded_value_token_position_mismatch={skipped_value_token_position_mismatch}, "
        f"discarded_post_filter_quota={skipped_post_filter_quota}"
    )
    if skipped_mismatched_span and patch_positions == "question_span":
        print(
            "run_noop_patching_cot_boundary: use --allow_truncated_span to patch "
            "the shared prefix for mismatched question-span token lengths."
        )


def _pair_prompts_and_gold(row, tokenizer, canonical_direction, pair_dataset):
    """Resolve (src_prompt, tgt_prompt, gold_answer) for one pair row.

    Centralises the direction-dependent question/answer flipping so the
    baselines code and any other caller see identical conventions.
    """
    if canonical_direction == "wrong_to_correct":
        src_q, tgt_q = row["question_w"], row["question_c"]
        gold = (
            str(row["answer_w"])
            if _uses_source_answer_as_gold(pair_dataset)
            else str(row["answer_c"])
        )
    else:
        src_q, tgt_q = row["question_c"], row["question_w"]
        gold = (
            str(row["answer_c"])
            if _uses_source_answer_as_gold(pair_dataset)
            else str(row["answer_w"])
        )
    src_prompt, _, _ = _apply_cot_template_with_question_span(src_q, tokenizer)
    tgt_prompt, _, _ = _apply_cot_template_with_question_span(tgt_q, tokenizer)
    return src_prompt, tgt_prompt, gold


def _pair_target_gold(row, canonical_direction, pair_dataset):
    """Resolve target-side gold for pair datasets where source and target
    have distinct correct answers (currently just gsm_sym_within_template).

    Returns None for pair datasets where the contrast doesn't have a meaningful
    target gold (e.g. noop_clean_self, where both prompts share the same
    underlying numerical answer).
    """
    if not _is_gsm_sym_within_template_dataset(pair_dataset):
        return None
    if canonical_direction == "wrong_to_correct":
        return str(row["answer_c"])
    return str(row["answer_w"])


def _compute_cot_boundary_baselines_dynamic_gold(
    model, tokenizer, pair_specs, out_path, max_new_tokens, overwrite,
    num_baseline_shards, baseline_shard,
):
    """Two-pass baseline computation for wrong-wrong (dynamic gold).

    Pass 1: generate each unique prompt; cache (gen_ids, gen_text, boundary
    log-softmax vector, parsed emitted answer).
    Pass 2: iterate pairs, set gold = source's parsed emitted answer, drop
    pairs where target's parsed answer == source's, and look up logprobs
    from the cached boundary log-softmax vectors. Single forward per
    unique prompt (no second model pass needed).
    """
    # Phase 1: generate each unique prompt and cache parsed answer + boundary lsm
    unique_prompts = []
    seen = set()
    for _, src_prompt, tgt_prompt, _, _ in pair_specs:
        for p in (src_prompt, tgt_prompt):
            if p not in seen:
                seen.add(p)
                unique_prompts.append(p)

    print(
        f"compute_cot_boundary_baselines (dynamic gold): {len(pair_specs)} "
        f"pairs to run, {len(unique_prompts)} unique prompts to generate "
        f"(dedup ratio {len(pair_specs) * 2 / max(len(unique_prompts), 1):.1f}× "
        f"vs naive 2 gens/pair)"
    )

    model.eval()
    device = next(model.parameters()).device

    prompt_info: dict[str, dict] = {}
    for prompt_text in tqdm(unique_prompts, desc="discover golds"):
        inputs = _encode(prompt_text, tokenizer, device)
        prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
        with torch.no_grad():
            gen_ids, gen_text, gen_scores = _greedy_generate(
                model, tokenizer, inputs,
                max_new_tokens=max_new_tokens, return_scores=True,
            )
        boundary_end = _boundary_end_index(tokenizer, gen_ids)
        boundary_lsm = None
        if boundary_end is not None and boundary_end < len(gen_scores):
            # Cache one log-softmax vector per prompt (~256 KB fp16 at vocab=128K).
            boundary_lsm = (
                F.log_softmax(gen_scores[boundary_end][0].float(), dim=-1)
                .detach().cpu().to(torch.float16)
            )
        prompt_info[prompt_text] = {
            "prompt_ids": prompt_ids,
            "gen_ids": gen_ids,
            "gen_text": gen_text,
            "boundary_end": boundary_end,
            "boundary_lsm": boundary_lsm,
            "emitted": _extract_emitted_answer(gen_text),
        }
        del gen_scores

    # Phase 2: per-pair gold assignment, degenerate filtering, write
    written = 0
    skipped_no_source_answer = 0
    skipped_no_source_boundary = 0
    skipped_degenerate = 0
    with open(out_path, "a" if not overwrite else "w") as f_out:
        for pair_idx, (key, src_prompt, tgt_prompt, _, _) in enumerate(pair_specs):
            src = prompt_info.get(src_prompt)
            tgt = prompt_info.get(tgt_prompt)
            if src is None or src["emitted"] is None:
                skipped_no_source_answer += 1
                continue
            gold = src["emitted"]
            if tgt is not None and _numbers_equal(tgt.get("emitted"), gold):
                skipped_degenerate += 1
                continue
            # Need a boundary lsm to score gold; if source's lsm is missing
            # the metric is unrecoverable for that pair.
            if src["boundary_lsm"] is None:
                skipped_no_source_boundary += 1
                continue
            gold_ans_id = _first_ans_token(tokenizer, gold)
            src_lp = float(src["boundary_lsm"][gold_ans_id])
            if tgt is not None and tgt["boundary_lsm"] is not None:
                tgt_lp = float(tgt["boundary_lsm"][gold_ans_id])
            else:
                tgt_lp = float("nan")
            src_correct = _is_correct_final_answer(src["gen_text"], gold)  # True by construction
            tgt_correct = (
                _is_correct_final_answer(tgt["gen_text"], gold) if tgt is not None else False
            )
            rec = {
                "original_id": int(key[0]),
                "instance_c": int(key[1]),
                "instance_w": int(key[2]),
                "gold_answer": str(gold),
                "unpatched_source_correct": bool(src_correct),
                "unpatched_source_gold_logprob": src_lp,
                "unpatched_target_correct": bool(tgt_correct),
                "unpatched_target_gold_logprob": tgt_lp,
            }
            f_out.write(json.dumps(rec) + "\n")
            f_out.flush()
            written += 1

    print(
        f"compute_cot_boundary_baselines (dynamic gold): "
        f"wrote {written} records, "
        f"skipped_no_source_answer={skipped_no_source_answer}, "
        f"skipped_no_source_boundary={skipped_no_source_boundary}, "
        f"skipped_degenerate={skipped_degenerate}, "
        f"out={out_path}"
    )


def compute_cot_boundary_baselines(model, tokenizer, model_id, model_name,
                                   max_pairs_per_orig=None, max_new_tokens=1024,
                                   direction="correct_to_wrong",
                                   pair_dataset="noop_clean_self",
                                   overwrite=False, allow_truncated_span=False,
                                   baseline_shard=0, num_baseline_shards=1,
                                   use_cached_cot=False):
    """Run unpatched source/target generations and cache per-pair baselines.

    Generations are deduped: each unique prompt is generated ONCE, then all
    (gold_answer) evaluations needed for any pair containing that prompt
    are computed from the cached generation. Within a single `original_id`
    a single `instance_c` (or `instance_w`) is typically reused across many
    pairs, so dedup is a large multiplicative win.

    The cached generation tensors (especially `scores`) are released after
    a prompt's evaluations are complete, so peak memory stays bounded.

    The baselines file is direction-specific (gold answer flips with direction)
    but is shared across all scopes (layer/mlp/attn_output/attn_head).

    Parallelism: when `num_baseline_shards > 1`, the pair list is partitioned
    by `pair_idx % num_baseline_shards == baseline_shard` and the output is
    written to a per-shard file `<baselines>.shard{i}of{n}.jsonl`. The
    baselines loader transparently merges all shard files alongside the
    canonical filename, so the layer-sweep code doesn't need to know about
    sharding.
    """
    if num_baseline_shards < 1:
        raise ValueError("num_baseline_shards must be >= 1")
    if not 0 <= baseline_shard < num_baseline_shards:
        raise ValueError(
            f"baseline_shard {baseline_shard} out of range for "
            f"num_baseline_shards {num_baseline_shards}"
        )
    if use_cached_cot and not _is_gsm_sym_within_template_dataset(pair_dataset):
        raise ValueError(
            "use_cached_cot=True is currently supported only for "
            "pair_dataset=gsm_sym_within_template (other datasets don't "
            "carry inference-time `original_cot_*` strings on pair rows)."
        )
    canonical_direction = _canonical_direction(direction)

    pairs = build_cot_boundary_pairs(
        tokenizer, model_id, max_pairs_per_orig=max_pairs_per_orig,
        pair_dataset=pair_dataset, experiment="cot_boundary",
    )
    if pairs.empty:
        print("No CoT boundary pairs found after filtering.")
        return

    span_tag = "_question_span" + ("_truncated" if allow_truncated_span else "")
    dir_tag = _direction_tag(direction, pair_dataset)
    canonical_out_path = _baselines_path(
        model_name, "cot_boundary", pair_dataset, dir_tag, span_tag=span_tag,
    )
    canonical_out_path.parent.mkdir(parents=True, exist_ok=True)
    if num_baseline_shards == 1:
        out_path = canonical_out_path
    else:
        out_path = canonical_out_path.with_name(
            f"{canonical_out_path.stem}.shard{baseline_shard}of{num_baseline_shards}"
            f"{canonical_out_path.suffix}"
        )

    # For resume: consider any record across all sibling shard files as "done"
    # so each shard skips work that any other shard has already completed. This
    # is conservative and avoids redundant generations even when shard
    # assignment changes between runs (e.g., re-shard from 4 to 8).
    done: set = set()
    if not overwrite:
        if num_baseline_shards > 1:
            done = set(_load_cot_boundary_baselines(canonical_out_path).keys())
        else:
            done = _load_done_keys(out_path)
        if done:
            print(
                f"Resuming baselines — {len(done)} pairs already cached "
                f"(across all shard files)"
            )

    # Resolve all pairs first; partition by shard; skip cached. This supports
    # resume and lets us enumerate the dedup work up front.
    # Tuple layout: (key, src_prompt, tgt_prompt, gold_answer, target_gold_answer,
    #                src_cot, tgt_cot). src_cot/tgt_cot are None unless
    #                use_cached_cot=True (within_template-only).
    pair_specs = []
    skipped_done = 0
    for pair_idx, (_, row) in enumerate(pairs.iterrows()):
        if pair_idx % num_baseline_shards != baseline_shard:
            continue
        key = (
            int(row["original_id"]),
            int(row["instance_c"]),
            int(row["instance_w"]),
        )
        if key in done:
            skipped_done += 1
            continue
        src_prompt, tgt_prompt, gold_answer = _pair_prompts_and_gold(
            row, tokenizer, canonical_direction, pair_dataset
        )
        target_gold = _pair_target_gold(row, canonical_direction, pair_dataset)
        src_cot = tgt_cot = None
        if use_cached_cot:
            if canonical_direction == "wrong_to_correct":
                src_cot = str(row.get("original_cot_w", ""))
                tgt_cot = str(row.get("original_cot_c", ""))
            else:
                src_cot = str(row.get("original_cot_c", ""))
                tgt_cot = str(row.get("original_cot_w", ""))
            if not src_cot or not tgt_cot:
                raise ValueError(
                    "use_cached_cot=True but pair "
                    f"{key} is missing original_cot_c/w. Either rebuild "
                    "pairs with the cached CoTs populated, or drop the flag."
                )
        pair_specs.append((key, src_prompt, tgt_prompt, gold_answer, target_gold,
                           src_cot, tgt_cot))

    if num_baseline_shards > 1:
        print(
            f"compute_cot_boundary_baselines: shard {baseline_shard}/"
            f"{num_baseline_shards}, this_shard_pair_count={len(pair_specs)}, "
            f"writing to {out_path.name}"
        )

    if not pair_specs:
        print(
            f"compute_cot_boundary_baselines: total_pairs={len(pairs)}, "
            f"skipped_already_done={skipped_done}, nothing to do."
        )
        return

    # ---- Dynamic gold path (e.g. noop_clean_self_ww) -------------------
    # For pair datasets where gold = source's *currently emitted* answer
    # (rather than a deterministic CSV field), we can't pre-bake gold in
    # the pair builder because greedy regen of high-entropy "wrong" prompts
    # produces a different wrong number than the eval CSV records ~47% of
    # the time. Discover gold from the model at baseline time instead, and
    # drop pairs that collapse to identical regenerated answers.
    if pair_dataset == "noop_clean_self_ww":
        return _compute_cot_boundary_baselines_dynamic_gold(
            model, tokenizer, pair_specs, out_path, max_new_tokens, overwrite,
            num_baseline_shards, baseline_shard,
        )

    # Map each unique prompt -> set of gold_answers it will be evaluated
    # against. Each prompt is generated once and evaluated against all needed
    # golds while its scores tensor is in memory.
    prompt_to_golds: dict[str, set[str]] = {}
    # For the cached-CoT path: associate each prompt with its (single) cached
    # CoT string. Conflict here would mean two pairs share a prompt text but
    # carry different CoTs — a data-integrity bug, so raise.
    prompt_to_cached_cot: dict[str, str] = {}
    for _, src_prompt, tgt_prompt, gold_answer, target_gold, src_cot, tgt_cot in pair_specs:
        prompt_to_golds.setdefault(src_prompt, set()).add(gold_answer)
        prompt_to_golds.setdefault(tgt_prompt, set()).add(gold_answer)
        # Within_template-style: also evaluate each unpatched run against
        # target's gold so the patched-side normalization has clean endpoints.
        if target_gold is not None and target_gold != gold_answer:
            prompt_to_golds[src_prompt].add(target_gold)
            prompt_to_golds[tgt_prompt].add(target_gold)
        if use_cached_cot:
            for prompt, cot in ((src_prompt, src_cot), (tgt_prompt, tgt_cot)):
                existing = prompt_to_cached_cot.get(prompt)
                if existing is not None and existing != cot:
                    raise ValueError(
                        "Two pairs share the same prompt text but carry "
                        "different cached CoTs — refusing to silently pick "
                        "one. Investigate the pair builder."
                    )
                prompt_to_cached_cot[prompt] = cot

    print(
        f"compute_cot_boundary_baselines: {len(pair_specs)} pairs to run, "
        f"{len(prompt_to_golds)} unique prompts to generate "
        f"(dedup ratio {len(pair_specs) * 2 / max(len(prompt_to_golds), 1):.1f}× "
        f"vs naive 2 gens/pair)"
    )

    # Reverse index + per-pair completion tracking so each pair can be written
    # to disk (and flushed) the moment both its src and tgt prompts are done.
    # This restores the per-pair resume granularity of the original
    # (pre-dedup) loop: a mid-job kill loses at most the work for prompts
    # generated since the most recently completed pair write.
    prompt_to_pending_pairs: dict[str, set[int]] = {}
    pair_completion: dict[int, dict] = {}
    for pair_idx, (key, src_prompt, tgt_prompt, gold_answer, target_gold,
                   _src_cot, _tgt_cot) in enumerate(pair_specs):
        prompt_to_pending_pairs.setdefault(src_prompt, set()).add(pair_idx)
        prompt_to_pending_pairs.setdefault(tgt_prompt, set()).add(pair_idx)
        pair_completion[pair_idx] = {
            "key": key,
            "gold_answer": gold_answer,
            "target_gold_answer": target_gold,
            "src_prompt": src_prompt,
            "tgt_prompt": tgt_prompt,
            "src_metrics": None,
            "tgt_metrics": None,
        }

    model.eval()
    device = next(model.parameters()).device

    written = 0
    desc = "cached-CoT forwards" if use_cached_cot else "unique gens"
    with open(out_path, "a" if not overwrite else "w") as f_out:
        for prompt_text, gold_answers in tqdm(
            prompt_to_golds.items(), desc=desc, total=len(prompt_to_golds)
        ):
            if use_cached_cot:
                cached_cot = prompt_to_cached_cot[prompt_text]
                per_gold_metrics = _score_cot_boundary_from_cached_cot(
                    model, tokenizer, prompt_text, cached_cot, gold_answers,
                )
            else:
                inputs = _encode(prompt_text, tokenizer, device)
                prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
                with torch.no_grad():
                    gen_ids, gen_text, gen_scores = _greedy_generate(
                        model, tokenizer, inputs,
                        max_new_tokens=max_new_tokens, return_scores=True,
                    )
                per_gold_metrics: dict[str, tuple[bool, float]] = {}
                for gold_answer in gold_answers:
                    correct = _is_correct_final_answer(gen_text, gold_answer)
                    logprob = _gold_answer_logprob_from_generation_scores(
                        model, tokenizer, prompt_ids, gen_ids, gold_answer, gen_scores,
                    )
                    per_gold_metrics[gold_answer] = (bool(correct), float(logprob))
                del gen_scores

            # Notify every pair waiting on this prompt. If both sides are now
            # complete, write the pair immediately and flush.
            for pair_idx in prompt_to_pending_pairs.pop(prompt_text, set()):
                spec = pair_completion[pair_idx]
                # Store the full per-gold metrics dict for this prompt so the
                # writer can pull both source-gold and target-gold readings.
                if spec["src_prompt"] == prompt_text and spec["src_metrics"] is None:
                    spec["src_metrics"] = per_gold_metrics
                if spec["tgt_prompt"] == prompt_text and spec["tgt_metrics"] is None:
                    spec["tgt_metrics"] = per_gold_metrics
                if spec["src_metrics"] is None or spec["tgt_metrics"] is None:
                    continue
                sg = spec["gold_answer"]
                tg = spec["target_gold_answer"]
                src_correct, src_logprob = spec["src_metrics"][sg]
                tgt_correct, tgt_logprob = spec["tgt_metrics"][sg]
                record = {
                    "original_id": spec["key"][0],
                    "instance_c": spec["key"][1],
                    "instance_w": spec["key"][2],
                    "gold_answer": sg,
                    "unpatched_source_correct": bool(src_correct),
                    "unpatched_source_gold_logprob": float(src_logprob),
                    "unpatched_target_correct": bool(tgt_correct),
                    "unpatched_target_gold_logprob": float(tgt_logprob),
                }
                if tg is not None and tg != sg:
                    src_tg_correct, src_tg_lp = spec["src_metrics"][tg]
                    tgt_tg_correct, tgt_tg_lp = spec["tgt_metrics"][tg]
                    record.update({
                        "target_gold_answer": tg,
                        "unpatched_source_target_gold_correct": bool(src_tg_correct),
                        "unpatched_source_target_gold_logprob": float(src_tg_lp),
                        "unpatched_target_target_gold_correct": bool(tgt_tg_correct),
                        "unpatched_target_target_gold_logprob": float(tgt_tg_lp),
                    })
                f_out.write(json.dumps(record) + "\n")
                f_out.flush()
                written += 1
                del pair_completion[pair_idx]

    if pair_completion:
        # Safety net: every pair should have been completed and written above.
        # If something is left, surface it loudly rather than silently dropping.
        print(
            f"WARNING: {len(pair_completion)} pairs left unwritten — likely a "
            "logic bug in the streaming-write path. Pair specs that survived: "
            f"{list(pair_completion.keys())[:5]}..."
        )

    print(
        f"compute_cot_boundary_baselines: total_pairs={len(pairs)}, "
        f"skipped_already_done={skipped_done}, unique_prompts_generated="
        f"{len(prompt_to_golds)}, written={written} -> {out_path}"
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_noop_patching(model_name, scope, layers=None, experiment="direct",
                       direction="correct_to_wrong", pair_dataset="noop_clean_self",
                       min_abs_denom=0.5, aggregate_by=None,
                       allow_truncated_span=False, patch_positions="prompt_end"):
    aggregate_by = aggregate_by or _default_aggregate_by(pair_dataset)
    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    if experiment == "cot_boundary":
        if patch_positions == "value_tokens":
            span_tag = "_value_tokens"
        else:
            span_tag = "_question_span" + ("_truncated" if allow_truncated_span else "")
    elif experiment == "direct" and patch_positions == "question_span":
        span_tag = "_question_span" + ("_truncated" if allow_truncated_span else "")
    else:
        span_tag = ""
    dir_tag = _direction_tag(direction, pair_dataset)
    filter_tag = _denom_filter_tag(experiment, min_abs_denom)
    aggregation_tag = _aggregation_tag(aggregate_by, pair_dataset)
    result_dir = _result_dir(pair_dataset)
    path = _result_path(
        model_name, experiment, scope, span_tag, layers_tag, dir_tag,
        pair_dataset,
    )
    if not path.exists():
        print(f"Results not found: {path}")
        return

    results = [json.loads(l) for l in open(path)]
    if not results:
        print(f"No results to plot in {path}")
        return
    n_raw = len(results)
    if experiment == "direct":
        results = _filter_by_direct_denom(results, min_abs_denom)
        filter_desc = f"absden <= {min_abs_denom}"
    elif experiment == "cot_boundary":
        results = _filter_by_cot_boundary_gap(results, min_abs_denom)
        filter_desc = f"|source_target_gap| <= {min_abs_denom}"
    else:
        filter_desc = None
    if filter_desc is not None and min_abs_denom and min_abs_denom > 0:
        if not results:
            print(
                f"No results left after absden > {min_abs_denom} filter in {path}"
            )
            return
        _log_filter(
            "plot_noop_patching", n_raw, len(results), filter_desc,
        )
    all_effects, n_units, n_rows = _aggregate_effects(results, aggregate_by)
    mean_effects = np.nanmean(all_effects, axis=0)             # (L[, H])
    ci_lo_effects, ci_hi_effects = _bootstrap_ci(all_effects)
    title_prefix = (
        "Noop-clean direct patching" if experiment == "direct" and pair_dataset == "noop_clean_self"
        else "Noop-clean CoT boundary patching" if experiment == "cot_boundary" and pair_dataset == "noop_clean_self"
        else f"{pair_dataset} direct patching" if experiment == "direct"
        else f"{pair_dataset} CoT boundary patching"
    )
    direction_title = _direction_title(direction, pair_dataset)
    y_label = (
        "Normalised patching effect\n(on answer logprob difference)"
        if experiment == "direct"
        else "Patched final-answer success rate"
    )

    if scope in ("layer", "mlp", "attn_output"):
        fig, ax = plt.subplots(figsize=(10, 4))
        # Block-output index i -> displayed label i+1 so the x-axis aligns
        # with template_similarity's "residual after L blocks" convention.
        xs = np.arange(mean_effects.shape[0]) + 1
        (line,) = ax.plot(xs, mean_effects[:, 0], lw=2)
        ax.fill_between(
            xs, ci_lo_effects[:, 0], ci_hi_effects[:, 0],
            color=line.get_color(), alpha=0.2, linewidth=0,
            label=f"95% bootstrap CI (N={n_units})",
        )
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{title_prefix} — {scope} — {direction_title}\n"
            f"{model_name}  |  N={n_units} {aggregate_by}"
            f"{f' ({n_rows} rows)' if aggregate_by == 'original_id' else ''}"
            f"{f'  |  absden>{min_abs_denom}' if experiment in ('direct', 'cot_boundary') and min_abs_denom else ''}",
            fontsize=12,
        )
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

    else:  # attn_head → heatmap
        # Slice to active rows so image coordinates match layer indices directly
        active = layers if layers else list(range(mean_effects.shape[0]))
        display = mean_effects[active, :]
        lim = np.nanpercentile(np.abs(display), 95) or 1e-9
        fig, ax = plt.subplots(figsize=(14, max(4, len(active) // 4)))
        im = ax.imshow(
            display, aspect="auto", interpolation="nearest",
            cmap="RdBu_r", vmin=-lim, vmax=lim,
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax, label=y_label)
        ax.set_yticks(range(len(active)))
        ax.set_yticklabels([a + 1 for a in active])
        ax.set_title(
            f"{title_prefix} — per-head ({scope}) — {direction_title}\n"
            f"{model_name}  |  N={n_units} {aggregate_by}"
            f"{f' ({n_rows} rows)' if aggregate_by == 'original_id' else ''}"
            f"{f'  |  absden>{min_abs_denom}' if experiment in ('direct', 'cot_boundary') and min_abs_denom else ''}",
            fontsize=12,
        )
        plt.tight_layout()

    save_path = _result_path(
        model_name, experiment, scope, span_tag, layers_tag, dir_tag,
        pair_dataset, extra_tag=f"{filter_tag}{aggregation_tag}", suffix=".png",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {save_path}")

    if experiment == "cot_boundary" and "gold_logprob_deltas" in results[0]:
        metric_key = (
            "normalized_gold_logprob_effects"
            if "normalized_gold_logprob_effects" in results[0]
            else "gold_logprob_deltas"
        )
        all_gold_metric, n_units_gold, n_rows_gold = _aggregate_effects(
            results, aggregate_by, key=metric_key,
        )
        mean_gold_metric = np.nanmean(all_gold_metric, axis=0)
        ci_lo_gold, ci_hi_gold = _bootstrap_ci(all_gold_metric)
        metric_label = (
            "Normalized patching effect\n(on source-answer logprob)"
            if metric_key == "normalized_gold_logprob_effects"
            else "Source-answer logprob delta\n(vs unpatched target)"
        )

        if scope in ("layer", "mlp", "attn_output"):
            fig, ax = plt.subplots(figsize=(10, 4))
            xs = np.arange(mean_gold_metric.shape[0]) + 1
            (line,) = ax.plot(xs, mean_gold_metric[:, 0], lw=2)
            ax.fill_between(
                xs, ci_lo_gold[:, 0], ci_hi_gold[:, 0],
                color=line.get_color(), alpha=0.2, linewidth=0,
                label=f"95% bootstrap CI (N={n_units_gold})",
            )
            ax.axhline(0, color="grey", lw=0.8)
            if metric_key == "normalized_gold_logprob_effects":
                ax.axhline(1, color="grey", lw=0.8, alpha=0.5)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label)
            ax.set_title(
                f"Noop CoT boundary patching — {scope}\n"
                f"{model_name}  |  N={n_units_gold} {aggregate_by}"
                f"{f' ({n_rows_gold} rows)' if aggregate_by == 'original_id' else ''}",
                fontsize=12,
            )
            ax.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
        else:
            active = layers if layers else list(range(mean_gold_metric.shape[0]))
            display = mean_gold_metric[active, :]
            lim = np.nanpercentile(np.abs(display), 95) or 1e-9
            fig, ax = plt.subplots(figsize=(14, max(4, len(active) // 4)))
            im = ax.imshow(
                display, aspect="auto", interpolation="nearest",
                cmap="RdBu_r", vmin=-lim, vmax=lim,
            )
            ax.set_xlabel("Head")
            ax.set_ylabel("Layer")
            plt.colorbar(im, ax=ax, label=metric_label)
            ax.set_yticks(range(len(active)))
            ax.set_yticklabels([a + 1 for a in active])
            ax.set_title(
                f"Noop CoT boundary patching — per-head ({scope})\n"
                f"{model_name}  |  N={n_units_gold} {aggregate_by}"
                f"{f' ({n_rows_gold} rows)' if aggregate_by == 'original_id' else ''}",
                fontsize=12,
            )
            plt.tight_layout()

        metric_tag = (
            "_normalized_effect"
            if metric_key == "normalized_gold_logprob_effects"
            else "_gold_lp_delta"
        )
        save_path = _result_path(
            model_name, experiment, scope, span_tag, layers_tag, dir_tag,
            pair_dataset, extra_tag=f"{filter_tag}{aggregation_tag}{metric_tag}",
            suffix=".png",
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot → {save_path}")


def _scope_compare_stage_boundaries(pair_dataset):
    """Stage-boundary markers to overlay on the scope-comparison plot.

    Picked per dataset so the boundaries match the stage(s) the experiment
    actually probes:
    - p1_vs_padded_*       → L22 abstraction, L36 symbolic formulation
    - gsm_sym_within_template → L36 formulation, L40 variable binding
    - noop / filler_vs_noop   → full four-stage scaffold (NoOp diagnosis)
    """
    if _is_p1_padded_dataset(pair_dataset):
        return [(22, "Abstraction"), (36, "Symbolic\nformulation")]
    if _is_gsm_sym_within_template_dataset(pair_dataset) or _is_noop_clean_self_ww_dataset(pair_dataset):
        return [(36, "Symbolic\nformulation"), (40, "Variable\nbinding")]
    return [
        (22, "Abstraction"),
        (36, "Symbolic\nformulation"),
        (40, "Variable\nbinding"),
    ]


def plot_noop_patching_scope_comparison(model_name, experiment="direct", layers=None,
                                        pair_dataset="noop_clean_self",
                                        min_abs_denom=0.5, aggregate_by=None):
    if experiment != "direct":
        print("Scope comparison plot is currently implemented for direct patching only.")
        return

    aggregate_by = aggregate_by or _default_aggregate_by(pair_dataset)
    scopes = ["layer", "attn_output", "mlp"]
    result_dir = _result_dir(pair_dataset)
    all_directions = (
        ["p1_to_padded", "padded_to_p1"]
        if _is_p1_padded_dataset(pair_dataset)
        else ["source_to_target", "target_to_source"]
        if _is_gsm_sym_within_template_dataset(pair_dataset) or _is_noop_clean_self_ww_dataset(pair_dataset)
        else ["correct_to_wrong", "wrong_to_correct"]
    )
    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    filter_tag = _denom_filter_tag(experiment, min_abs_denom)
    aggregation_tag = _aggregation_tag(aggregate_by, pair_dataset)
    # Headline palette: layer-blue / attn-green / mlp-red, matched to the
    # template_similarity and cot_swap figures.
    colors = {
        "layer": "#2166ac",
        "attn_output": "#1b7837",
        "mlp": "#d6604d",
    }
    labels = {
        "layer": "Full layer residual",
        "attn_output": "Attention output",
        "mlp": "MLP output",
    }

    series = {}
    ci_bands = {}
    y_max = None
    y_min = None
    sample_sizes = {}
    available_directions = []
    for direction in all_directions:
        dir_series = {}
        dir_ci = {}
        dir_sizes = {}
        ok = True
        for scope in scopes:
            path = _result_path(
                model_name, experiment, scope, "", layers_tag,
                _direction_tag(direction, pair_dataset), pair_dataset,
            )
            if not path.exists():
                print(f"Results not found (skipping {direction}): {path}")
                ok = False
                break
            results = [json.loads(line) for line in open(path) if line.strip()]
            if not results:
                print(f"No results in {path} (skipping {direction})")
                ok = False
                break
            n_raw = len(results)
            results = _filter_by_direct_denom(results, min_abs_denom)
            if not results:
                print(
                    f"No results left after absden > {min_abs_denom} "
                    f"filter in {path} (from {n_raw})"
                )
                ok = False
                break
            _log_filter(
                "plot_noop_patching_scope_comparison",
                n_raw,
                len(results),
                f"absden <= {min_abs_denom} for {direction}/{scope}",
            )
            all_effects, n_units, n_rows = _aggregate_effects(results, aggregate_by)
            mean_effects = np.nanmean(all_effects, axis=0)[:, 0]
            ci_lo, ci_hi = _bootstrap_ci(all_effects)
            dir_series[scope] = mean_effects
            dir_ci[scope] = (ci_lo[:, 0], ci_hi[:, 0])
            dir_sizes[scope] = (n_units, n_rows)
        if ok:
            available_directions.append(direction)
            for scope in scopes:
                series[(direction, scope)] = dir_series[scope]
                ci_bands[(direction, scope)] = dir_ci[scope]
                sample_sizes[(direction, scope)] = dir_sizes[scope]
                lo, hi = dir_ci[scope]
                scope_max = np.nanmax(hi)
                scope_min = np.nanmin(lo)
                y_max = scope_max if y_max is None else max(y_max, scope_max)
                y_min = scope_min if y_min is None else min(y_min, scope_min)

    if not available_directions:
        print("No valid scope comparison data found.")
        return

    pad = 0.05 * max(abs(y_min), abs(y_max), 1e-9)
    n_panels = len(available_directions)
    # Headline aspect: 12×4.8 single, scale width for the side-by-side variant.
    fig_width = 12.0 if n_panels == 1 else 6.5 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, 4.5),
                             sharey=n_panels > 1)
    if n_panels == 1:
        axes = [axes]

    num_layers = len(series[(available_directions[0], "layer")])
    stage_boundaries = _scope_compare_stage_boundaries(pair_dataset)

    for ax, direction in zip(axes, available_directions):
        for scope in scopes:
            arr = series[(direction, scope)]
            lo, hi = ci_bands[(direction, scope)]
            xs = np.arange(len(arr)) + 1
            ax.fill_between(
                xs, lo, hi,
                color=colors[scope], alpha=0.18, linewidth=0,
            )
            ax.plot(xs, arr, lw=1.8, color=colors[scope], label=labels[scope])
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(_direction_title(direction, pair_dataset), fontsize=11)
        ax.set_ylim(y_min - pad, y_max + pad)
        _style_layer_axis(ax, num_layers, stage_boundaries=stage_boundaries)

    n_units, n_rows = sample_sizes[(available_directions[0], "layer")]
    axes[0].set_ylabel("Normalised patching effect\n(on answer log-prob difference)")
    axes[-1].legend(fontsize=9.5, loc="best", frameon=True)

    # No suptitle (matches headline plots); sample-size + filter info goes in
    # a small footer so it stays readable when the figure is embedded.
    unit_label = "templates" if aggregate_by == "original_id" else "pairs"
    footer = f"{model_name}  |  N = {n_units} {unit_label}"
    if aggregate_by == "original_id":
        footer += f" ({n_rows} rows)"
    if min_abs_denom:
        footer += f"  |  |Δ log P| > {min_abs_denom}"
    fig.text(0.5, -0.02, footer, ha="center", va="top", fontsize=9,
             color="#555555")
    plt.tight_layout()

    save_path = _scope_compare_path(
        model_name, experiment, layers_tag, filter_tag, aggregation_tag,
        pair_dataset,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Plot → {save_path}")


def _scope_compare_plot_path(model_name, experiment="direct", layers=None,
                             pair_dataset="noop_clean_self", min_abs_denom=0.5,
                             aggregate_by=None):
    aggregate_by = aggregate_by or _default_aggregate_by(pair_dataset)
    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    filter_tag = _denom_filter_tag(experiment, min_abs_denom)
    aggregation_tag = _aggregation_tag(aggregate_by, pair_dataset)
    return _scope_compare_path(
        model_name, experiment, layers_tag, filter_tag, aggregation_tag,
        pair_dataset,
    )


def maybe_regenerate_scope_comparison(model_name, experiment="direct", layers=None,
                                      pair_dataset="noop_clean_self",
                                      min_abs_denom=0.5, aggregate_by=None):
    if experiment != "direct":
        return
    aggregate_by = aggregate_by or _default_aggregate_by(pair_dataset)

    scopes = ["layer", "attn_output", "mlp"]
    directions = (
        ["p1_to_padded", "padded_to_p1"]
        if _is_p1_padded_dataset(pair_dataset)
        else ["source_to_target", "target_to_source"]
        if _is_gsm_sym_within_template_dataset(pair_dataset) or _is_noop_clean_self_ww_dataset(pair_dataset)
        else ["correct_to_wrong", "wrong_to_correct"]
    )
    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    result_dir = _result_dir(pair_dataset)
    compare_path = _scope_compare_plot_path(
        model_name, experiment=experiment, layers=layers,
        pair_dataset=pair_dataset, min_abs_denom=min_abs_denom,
        aggregate_by=aggregate_by,
    )

    component_paths = []
    for direction in directions:
        direction_paths = [
            _result_path(
                model_name, experiment, scope, "", layers_tag,
                _direction_tag(direction, pair_dataset), pair_dataset,
            )
            for scope in scopes
        ]
        if all(path.exists() for path in direction_paths):
            component_paths.extend(direction_paths)

    if not component_paths:
        print("Scope comparison not regenerated: component JSONLs not complete yet.")
        return

    latest_component_mtime = max(path.stat().st_mtime for path in component_paths)
    if compare_path.exists() and compare_path.stat().st_mtime >= latest_component_mtime:
        print(f"Scope comparison is up to date: {compare_path}")
        return

    print("Regenerating scope comparison from updated component JSONLs.")
    plot_noop_patching_scope_comparison(
        model_name, experiment=experiment, layers=layers,
        pair_dataset=pair_dataset, min_abs_denom=min_abs_denom,
        aggregate_by=aggregate_by,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="meta-llama/Llama-3.3-70B-Instruct",
                        choices=list(MODEL_NAME_MAP.keys()))
    parser.add_argument("--experiment", type=str,
                        choices=["direct", "cot_boundary"],
                        default="direct")
    parser.add_argument("--scope", type=str,
                        choices=["layer", "mlp", "attn_output", "attn_head"],
                        default="layer")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Restrict patching to these layers (for attn_head scope).")
    parser.add_argument("--max_pairs_per_orig", type=int, default=None,
                        help="Cap pairs per original_id (default: use all aligned pairs)")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Only used for cot_boundary: max tokens to generate.")
    parser.add_argument(
        "--allow_truncated_span",
        action="store_true",
        help=(
            "Only used for cot_boundary question-span patching: when source and "
            "target question spans have different token counts, patch the shared "
            "prefix instead of skipping the pair."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Pairs per forward pass for the direct experiment.")
    parser.add_argument("--direction", type=str,
                        choices=[
                            "correct_to_wrong", "wrong_to_correct",
                            "p1_to_padded", "padded_to_p1",
                            "source_to_target", "target_to_source",
                        ],
                        default="wrong_to_correct",
                        help=(
                            "Patch direction. For P1/padded datasets, prefer "
                            "p1_to_padded or padded_to_p1. For GSM-Symbolic "
                            "within-template, prefer source_to_target or "
                            "target_to_source. Legacy noop aliases correct_to_wrong "
                            "and wrong_to_correct are still accepted. Default "
                            "(wrong_to_correct) matches the shell wrappers; it "
                            "asks whether the noop-wrong run is recoverable by "
                            "patching from the correct activations."
                        ))
    parser.add_argument("--pair_dataset", type=str,
                        choices=list(PAIR_DATASET_PATHS.keys()),
                        default="noop_clean_self",
                        help="Which paired dataset to patch over. Use filler_vs_noop_tfm for transformers-direct-based pairs.")
    parser.add_argument("--scope_compare", action="store_true",
                        help="Plot layer/attn_output/mlp together with one panel per direction.")
    parser.add_argument("--min_abs_denom", type=float, default=0.5,
                        help="For direct plots, keep rows with abs(source_lp - target_lp) above this threshold.")
    parser.add_argument("--aggregate_by", choices=["auto", "row", "original_id"],
                        default="auto",
                        help="Plot aggregation. auto uses original_id for P1/padded and row otherwise.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing results instead of resuming.")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--baselines_only", action="store_true",
                        help="Only run unpatched source/target generations and write "
                             "the per-pair baseline jsonl. Layer-sweep tasks then reuse "
                             "these instead of regenerating per shard (~3x speedup). "
                             "Currently supported for --experiment cot_boundary.")
    parser.add_argument("--baseline_shard", type=int, default=0,
                        help="Which baseline shard this task owns (0-indexed). "
                             "Used with --num_baseline_shards to parallelize the "
                             "baselines computation across SLURM array tasks.")
    parser.add_argument("--num_baseline_shards", type=int, default=1,
                        help="Total number of baseline shards. Each baselines task "
                             "processes pairs where pair_idx %% num_baseline_shards == "
                             "baseline_shard and writes to a per-shard file. The "
                             "baselines loader merges all shard files transparently, "
                             "so no post-merge step is needed.")
    parser.add_argument("--allow_no_baseline_cache", action="store_true",
                        help="By default, cot_boundary patching fails fast if no "
                             "baseline cache is found, because each layer-shard "
                             "would otherwise re-generate the unpatched source and "
                             "target CoTs per pair (extremely expensive). Pass this "
                             "flag to explicitly accept that cost and run with "
                             "inline baseline computation.")
    parser.add_argument("--baseline_from_cached_cot", action="store_true",
                        help="For --baselines_only on gsm_sym_within_template: skip "
                             "autoregressive greedy generation and instead score "
                             "the cached `original_cot_c`/`original_cot_w` (already "
                             "produced under the same transformers-direct greedy "
                             "convention) at its #### boundary via a single forward "
                             "pass. Mathematically equivalent to live generation; "
                             "much cheaper. Errors out if cached CoTs are missing.")
    parser.add_argument("--value_tokens_position_exact", action="store_true",
                        help="For --patch_positions value_tokens: restrict to pairs "
                             "where source and target prompts tokenize to the same "
                             "length AND every operand sits at identical absolute "
                             "positions on both sides. Makes value_tokens patching "
                             "a pure-content intervention (no positional-encoding "
                             "confound). Results write to a sibling `value_tokens_"
                             "position_exact` directory so they don't overwrite the "
                             "looser filter's outputs.")
    parser.add_argument("--max_pairs_per_orig_post_filter", type=int, default=None,
                        help="For --patch_positions value_tokens: per-template cap "
                             "applied AFTER alignment + position-exact filters. "
                             "Bounds the per-template contribution to the patched "
                             "set; e.g. `--max_pairs_per_orig_post_filter 5` keeps "
                             "at most 5 actually-patched pairs per template. None "
                             "(default) = no cap.")
    parser.add_argument("--no_plot_after_run", action="store_true",
                        help="Write JSONL results only; skip per-run plot generation.")
    parser.add_argument("--patch_positions",
                        choices=["prompt_end", "question_span", "value_tokens"],
                        default="prompt_end",
                        help="For --experiment direct: prompt_end (default) patches "
                             "the last prompt token at each layer; question_span "
                             "patches every token in the aligned question span. "
                             "For --experiment cot_boundary: question_span (the "
                             "default in run_noop_patching_cot_boundary) or "
                             "value_tokens (patch only operand-value tokens, "
                             "aligned by variable name; requires "
                             "--pair_dataset gsm_sym_within_template).")
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]
    _validate_pair_dataset(args.pair_dataset, args.experiment)
    aggregate_by = (
        _default_aggregate_by(args.pair_dataset)
        if args.aggregate_by == "auto"
        else args.aggregate_by
    )

    if args.scope_compare:
        plot_noop_patching_scope_comparison(
            model_name, experiment=args.experiment, layers=args.layers,
            pair_dataset=args.pair_dataset,
            min_abs_denom=args.min_abs_denom,
            aggregate_by=aggregate_by,
        )
    elif args.plot_only:
        for scope in ["layer", "mlp", "attn_output", "attn_head"]:
            plot_noop_patching(
                model_name, scope, args.layers, experiment=args.experiment,
                direction=args.direction, pair_dataset=args.pair_dataset,
                min_abs_denom=args.min_abs_denom,
                aggregate_by=aggregate_by,
                allow_truncated_span=args.allow_truncated_span,
                patch_positions=args.patch_positions,
            )
        plot_noop_patching_scope_comparison(
            model_name, experiment=args.experiment, layers=args.layers,
            pair_dataset=args.pair_dataset,
            min_abs_denom=args.min_abs_denom,
            aggregate_by=aggregate_by,
        )
    elif args.baselines_only:
        if args.experiment != "cot_boundary":
            raise SystemExit(
                "--baselines_only is only supported for --experiment cot_boundary"
            )
        model, tokenizer = load_model(args.model_id)
        compute_cot_boundary_baselines(
            model, tokenizer, args.model_id, model_name,
            max_pairs_per_orig=args.max_pairs_per_orig,
            max_new_tokens=args.max_new_tokens,
            direction=args.direction,
            pair_dataset=args.pair_dataset,
            overwrite=args.overwrite,
            allow_truncated_span=args.allow_truncated_span,
            baseline_shard=args.baseline_shard,
            num_baseline_shards=args.num_baseline_shards,
            use_cached_cot=args.baseline_from_cached_cot,
        )
    else:
        model, tokenizer = load_model(args.model_id)
        if args.experiment == "direct":
            run_noop_patching_direct(
                model, tokenizer, args.model_id, model_name,
                scope=args.scope,
                max_pairs_per_orig=args.max_pairs_per_orig,
                layers=args.layers,
                batch_size=args.batch_size,
                direction=args.direction,
                pair_dataset=args.pair_dataset,
                overwrite=args.overwrite,
                patch_positions=args.patch_positions,
                allow_truncated_span=args.allow_truncated_span,
            )
        else:
            # CLI default ("prompt_end") only makes sense for --experiment direct.
            # For cot_boundary, fall back to question_span unless the user
            # explicitly asked for value_tokens.
            cot_patch_positions = (
                "value_tokens" if args.patch_positions == "value_tokens"
                else "question_span"
            )
            run_noop_patching_cot_boundary(
                model, tokenizer, args.model_id, model_name,
                scope=args.scope,
                max_pairs_per_orig=args.max_pairs_per_orig,
                layers=args.layers,
                max_new_tokens=args.max_new_tokens,
                direction=args.direction,
                pair_dataset=args.pair_dataset,
                overwrite=args.overwrite,
                allow_truncated_span=args.allow_truncated_span,
                allow_no_baseline_cache=args.allow_no_baseline_cache,
                patch_positions=cot_patch_positions,
                value_tokens_position_exact=args.value_tokens_position_exact,
                max_pairs_per_orig_post_filter=args.max_pairs_per_orig_post_filter,
            )
        # Auto-skip post-run plotting when this was a per-layer shard:
        # each shard's arrays have only one finite layer index, so a per-shard
        # plot is degenerate. Merge shards first (merge_noop_layers.py), then
        # run with --plot_only to produce the headline figure.
        suppress_plot = args.no_plot_after_run or bool(args.layers)
        if args.layers and not args.no_plot_after_run:
            layers_str = "-".join(str(L) for L in args.layers)
            print(
                f"Layer-sharded run for --layers {layers_str}; skipping post-run "
                "plot. Merge the layer shards, then rerun with --plot_only."
            )
        if not suppress_plot:
            plot_noop_patching(
                model_name, args.scope, args.layers, experiment=args.experiment,
                direction=args.direction, pair_dataset=args.pair_dataset,
                min_abs_denom=args.min_abs_denom,
                aggregate_by=aggregate_by,
                allow_truncated_span=args.allow_truncated_span,
                patch_positions=args.patch_positions,
            )
        if (
            not suppress_plot
            and args.experiment == "direct"
            and args.scope in ("layer", "mlp", "attn_output")
        ):
            maybe_regenerate_scope_comparison(
                model_name, experiment=args.experiment, layers=args.layers,
                pair_dataset=args.pair_dataset,
                min_abs_denom=args.min_abs_denom,
                aggregate_by=aggregate_by,
            )
