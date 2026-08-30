"""
CoT-swap logit lens experiments.

For each case where the model was given [noop_question + correct sym CoT trace] but
still generated distractor content (swap_correct=False):

1. Generate from [question + sym_reasoning] until "####" is produced.
2. Extend inputs to the "#### " position (last token = " " after "####").
3. Apply logit lens at each layer at that position.
Target: sym answer integer token.

Both runs are measured at the same token type (" " after "####"), reached via their
natural generation paths.

Usage:
    python run_cot_swap_logit_lens.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_cot_swap_logit_lens.py --model_id meta-llama/Llama-3.3-70B-Instruct --plot_only
    python run_cot_swap_logit_lens.py --contrast p1_vs_symbolic
    python run_cot_swap_logit_lens.py --contrast p1_vs_symbolic --padded_dataset all --p1_value_subset with_value
    python run_cot_swap_logit_lens.py --contrast p1_vs_padded_symbolic --padded_dataset delta0
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from utils.activation_patching_utils import tokenize
from utils.shared_utils import untuple
from config import (
    CACHE_DIR,
    ACTIVATION_PATCHING_RESULT_DIR,
    MODEL_NAME_MAP,
    COT_INSTRUCTION,
)
from utils.noop_utils import load_model, to_serializable, load_cot_swap_data
from utils.logit_lens_utils import get_logit_lens, forward_with_hooks


def extend_to_answer_position(model, tokenizer, inputs, reasoning,
                               tok_hash, tok_space, max_new_tokens=150):
    """Generate from inputs until '####' is produced, return inputs extended to '#### '."""
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0,
        )
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output[0, input_len:].tolist()

    for i, tid in enumerate(gen_ids):
        if tid == tok_hash:
            prefix = gen_ids[:i + 1]
            if i + 1 < len(gen_ids) and gen_ids[i + 1] == tok_space:
                prefix.append(tok_space)
            else:
                prefix.append(tok_space)
            device = inputs["input_ids"].device
            new_ids  = torch.cat([inputs["input_ids"],
                                   torch.tensor([prefix], device=device)], dim=1)
            new_mask = torch.cat([inputs["attention_mask"],
                                   torch.ones(1, len(prefix), device=device)], dim=1)
            generated_text = tokenizer.decode(gen_ids[:i], skip_special_tokens=True)
            return {"input_ids": new_ids, "attention_mask": new_mask}, generated_text

    return None, None

ROOT     = Path(__file__).resolve().parent
OUT_DIR  = Path(ACTIVATION_PATCHING_RESULT_DIR).parent / "cot_swap_logit_lens"
P1_MATCHED_PAIRS_PATH = ROOT / "data" / "gsm_symbolic" / "gsm_symbolic_vs_p1_length_matched_pairs.csv"
TFM_DIR = ROOT / "results" / "disentangled_evaluation" / "transformers_direct"
PADDED_TFM_DATASETS = {
    # key -> (inference_subdir, dataset_prefix) | dict for legacy
    # gsm_padded_symbolic: entity-anchored, digit-free padded build whose
    #   (oid, inst) inherits gsm_symbolic's variable bindings. Not entity-
    #   aligned to gsm_p1[oid, inst]; suitable for sym↔padded_sym contrasts,
    #   NOT for patching into p1.
    # gsm_padded_symbolic_p1_aligned: same clause strategy, but each (oid,
    #   inst) is built from the matched-pairs CSV's source_question_unpadded
    #   (rendered with p1's bindings) and length-matched to p1[oid, inst].
    #   Entity-aligned to gsm_p1[oid, inst]; use for p1_vs_padded_symbolic.
    # legacy_symbolic_padded: pre-filtered legacy inference runs under
    #   gsm_symbolic/. --length_filter selects which file (delta0 → exact-
    #   length match, lt2 → |delta|<2). Sym-bound (not entity-aligned to p1)
    #   — same semantics as gsm_padded_symbolic, preserved for back-compat
    #   with existing JSONL outputs.
    "gsm_padded_symbolic": ("gsm_padded_symbolic", "gsm_padded_symbolic"),
    "gsm_padded_symbolic_p1_aligned": (
        "gsm_padded_symbolic_p1_aligned", "gsm_padded_symbolic_p1_aligned",
    ),
    "legacy_symbolic_padded": {
        "delta0": ("gsm_symbolic", "gsm_symbolic_padded_to_p1_len_delta_0"),
        "lt2":    ("gsm_symbolic", "gsm_symbolic_padded_to_p1_len_delta_abs_lt2"),
    },
}

LENGTH_FILTERS = ("all", "delta0", "lt2")


def build_chat_inputs(question, model, tokenizer, instruction=COT_INSTRUCTION, answer_prefix=""):
    messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt = prompt + answer_prefix
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)


def cot_prefix_before_hash(reasoning):
    """Return the saved CoT text before the first #### marker."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    if "####" not in reasoning:
        return None
    return reasoning.split("####", 1)[0]


def get_token_logprob_lens(state_dict, model, target_token_id):
    """Return layerwise log p(target_token_id) at the final input position."""
    num_layers = model.config.num_hidden_layers
    result = np.full(num_layers, float("nan"))

    with torch.no_grad():
        for i in range(num_layers):
            key = f"layer_{i}"
            if key not in state_dict:
                continue
            h = untuple(state_dict[key])[:, -1, :]
            logits = model.lm_head(model.model.norm(h))
            result[i] = F.log_softmax(logits.float(), dim=-1)[0, target_token_id].item()
    return result


def _coerce_bool(series):
    return series.map(
        {
            True: True,
            False: False,
            "True": True,
            "False": False,
            "true": True,
            "false": False,
            1: True,
            0: False,
            1.0: True,
            0.0: False,
        }
    ).astype("boolean")


def _csv_model_aliases(model_name: str):
    """Return candidate model-name strings to try when looking up
    inference CSV filenames. The primary value is the MODEL_NAME_MAP
    short form (legacy convention); we also reverse-lookup the original
    HF model-id suffix because run_inference_transformers_direct.py
    actually writes CSVs with that case-preserved suffix."""
    aliases = [model_name]
    for hf_id, short in MODEL_NAME_MAP.items():
        if short == model_name:
            aliases.append(hf_id.split("/")[-1])
    # De-dup, preserve order.
    seen = set()
    out = []
    for a in aliases:
        if a not in seen:
            seen.add(a); out.append(a)
    return out


def _load_split_results(result_subdir, dataset_name, model_name):
    directory = TFM_DIR / result_subdir
    last_pattern = None
    for alias in _csv_model_aliases(model_name):
        last_pattern = f"{dataset_name}_set_*_results_{alias}_transformers.csv"
        files = sorted(directory.glob(last_pattern))
        if files:
            break
    else:
        files = []
    if not files:
        raise FileNotFoundError(
            f"No transformer-backend result files matched "
            f"{directory / last_pattern} (also tried aliases: "
            f"{_csv_model_aliases(model_name)})"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["split"] = int(str(path).split("_set_")[1].split("_")[0])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _filter_subset(df, length_filter):
    if length_filter == "delta0":
        return df[df["len_delta_after"].astype(int) == 0].copy()
    if length_filter == "lt2":
        return df[df["len_delta_after"].astype(int).abs() < 2].copy()
    if length_filter == "all":
        return df.copy()
    raise ValueError(f"Unknown length_filter: {length_filter}")


_TOKENIZER_CACHE: dict = {}


_SHORT_NAME_TO_HF_ID = {short: hf_id for hf_id, short in MODEL_NAME_MAP.items()}
# Inference CSVs are named with the model basename (the part after "/").
# For HF orgs whose basename != short name (e.g. "Qwen2.5-14B-Instruct" vs
# "qwen-14b-instruct"), this lets us resolve from basename → HF repo id.
_BASENAME_TO_HF_ID = {hf_id.split("/")[-1]: hf_id for hf_id in MODEL_NAME_MAP.keys()}


def _get_tokenizer_for_len_delta(model_name):
    """Return a tokenizer used solely to compute len_delta_after at load time.

    Loaded once per process; uses the model's HF tokenizer so token counts
    match what the patching runner sees."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    from transformers import AutoTokenizer
    # model_name may be either an HF repo id ("meta-llama/Llama-3.3-70B-Instruct"),
    # the short lowercase nickname from MODEL_NAME_MAP ("llama-3.3-70b-instruct"),
    # or the model basename used in inference filenames ("Qwen2.5-14B-Instruct").
    # The HF cache is keyed on the exact repo id; resolve to that.
    if "/" in model_name:
        hf_id = model_name
    elif model_name in _SHORT_NAME_TO_HF_ID:
        hf_id = _SHORT_NAME_TO_HF_ID[model_name]
    elif model_name in _BASENAME_TO_HF_ID:
        hf_id = _BASENAME_TO_HF_ID[model_name]
    else:
        hf_id = f"meta-llama/{model_name}"
    try:
        tok = AutoTokenizer.from_pretrained(hf_id, cache_dir=CACHE_DIR)
    except Exception:
        tok = AutoTokenizer.from_pretrained(hf_id)
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def _compute_len_delta_column(source_questions, target_questions, model_name):
    """Per-row len_delta_after = tok_len(source) - tok_len(target).

    Matches the construction convention in create_padded_symbolic.py: target
    is gsm_p1's question, source is the padded sym question. delta=0 means
    exact length match."""
    tok = _get_tokenizer_for_len_delta(model_name)
    def L(q):
        return len(tok(q, add_special_tokens=False)["input_ids"])
    src_lens = [L(q) for q in source_questions]
    tgt_lens = [L(q) for q in target_questions]
    return [s - t for s, t in zip(src_lens, tgt_lens)]


def _filter_distinct_answers(df, source_col="source_answer", target_col="target_answer"):
    source_answer_num = pd.to_numeric(df[source_col], errors="coerce")
    target_answer_num = pd.to_numeric(df[target_col], errors="coerce")
    numeric_same = (
        source_answer_num.notna()
        & target_answer_num.notna()
        & np.isclose(source_answer_num, target_answer_num)
    )
    string_same = df[source_col].astype(str) == df[target_col].astype(str)
    return df[~(numeric_same | string_same)].copy()


def _parse_p1_introduced_values(raw):
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [it.get("value") for it in items if it.get("value") is not None]


def _classify_p1_value_presence(padded_question, p1_values):
    if not p1_values:
        return "no_p1_value"
    flags = [str(v) in str(padded_question) for v in p1_values]
    if all(flags):
        return "all"
    if any(flags):
        return "some"
    return "none"


_SUBSET_TO_CLASS = {
    "no_value": {"no_p1_value"},
    "with_value": {"all", "some", "none"},
    "value_present": {"all"},
    "value_absent": {"none"},
}


def _apply_p1_value_subset(df, subsets, padded_question_col):
    if isinstance(subsets, str):
        subsets = [subsets]
    subsets = list(dict.fromkeys(subsets))  # de-dup, preserve order
    if "all" in subsets:
        return df
    if not subsets:
        raise ValueError("p1_value_subset must contain at least one subset")
    if "p1_introduced_numbers" not in df.columns:
        raise KeyError("p1_introduced_numbers column missing from matched-pairs CSV")

    allowed = set()
    for s in subsets:
        if s not in _SUBSET_TO_CLASS:
            raise ValueError(f"Unknown p1_value_subset: {s}")
        allowed |= _SUBSET_TO_CLASS[s]

    p1_vals = df["p1_introduced_numbers"].apply(_parse_p1_introduced_values)
    cls = pd.Series(
        [
            _classify_p1_value_presence(q, vs)
            for q, vs in zip(df[padded_question_col], p1_vals)
        ],
        index=df.index,
    )
    return df[cls.isin(allowed)].copy()


def _normalize_subsets(p1_value_subset):
    if isinstance(p1_value_subset, str):
        subsets = [p1_value_subset]
    else:
        subsets = list(p1_value_subset)
    subsets = list(dict.fromkeys(subsets))
    if not subsets:
        subsets = ["all"]
    if "all" in subsets:
        subsets = ["all"]
    return subsets


def _subset_tag(subsets):
    subsets = _normalize_subsets(subsets)
    if subsets == ["all"]:
        return ""
    return "_p1val_" + "+".join(subsets)


def _load_matched_metadata(
    length_filter, p1_value_subset="all", p1_value_question_col="source_question"
):
    if not P1_MATCHED_PAIRS_PATH.exists():
        raise FileNotFoundError(P1_MATCHED_PAIRS_PATH)

    df = pd.read_csv(P1_MATCHED_PAIRS_PATH)
    if p1_value_question_col not in df.columns:
        raise KeyError(
            f"{p1_value_question_col} column missing from matched-pairs CSV"
        )
    matched = df["matched"]
    if matched.dtype == object:
        matched = matched.astype(str).str.lower().map({"true": True, "false": False})
    df = df[matched.fillna(False).astype(bool)].copy()
    df = _filter_subset(df, length_filter)
    df = df.dropna(subset=["target_question", "source_answer", "target_answer"])
    df = _filter_distinct_answers(df)
    df = _apply_p1_value_subset(
        df, p1_value_subset, padded_question_col=p1_value_question_col
    )
    return df


def _load_p1_contrast_data(contrast, padded_dataset, model_name, p1_value_subset="all",
                            max_pairs_per_template=None, seed=0,
                            correctness_mode="cot", length_filter="all"):
    """Load p1-contrast data.

    correctness_mode: "cot" or "direct". Filters both sides by the matching
    `original_{mode}_correctness` column.
    length_filter: "all" | "delta0" | "lt2". Applied to len_delta_after =
    tok_len(source.question) - tok_len(target.question). For
    legacy_symbolic_padded, selects which pre-filtered inference CSV to load
    (delta0 → gsm_symbolic_padded_to_p1_len_delta_0; lt2 → ...len_delta_abs_lt2).
    """
    if correctness_mode not in {"cot", "direct"}:
        raise ValueError(f"Unknown correctness_mode: {correctness_mode}")
    if length_filter not in LENGTH_FILTERS:
        raise ValueError(f"Unknown length_filter: {length_filter}; "
                         f"choose one of {LENGTH_FILTERS}")
    correctness_col = f"original_{correctness_mode}_correctness"
    subsets = _normalize_subsets(p1_value_subset)
    if contrast == "p1_vs_symbolic":
        if padded_dataset != "all":
            raise ValueError(
                "p1_vs_symbolic is independent of padded length; use --padded_dataset all."
            )
        # gsm_symbolic_to_p1 and gsm_p1 share instantiations: for each
        # (original_id, instance), gsm_symbolic_to_p1.target_question == gsm_p1.question.
        # So we can deterministically pair on (original_id, instance) and require both
        # sides to have a correct CoT.
        source = _load_split_results("gsm_symbolic", "gsm_symbolic_to_p1", model_name)
        p1     = _load_split_results("gsm_p1", "gsm_p1", model_name)

        source[correctness_col] = _coerce_bool(source[correctness_col])
        p1[correctness_col]     = _coerce_bool(p1[correctness_col])
        source = source[source[correctness_col] == True].copy()
        p1     = p1[p1[correctness_col] == True].copy()

        merged = source[["original_id", "instance", "question", "answer", "original_cot"]].merge(
            p1[["original_id", "instance", "question", "answer"]],
            on=["original_id", "instance"],
            suffixes=("_source", "_target"),
            validate="one_to_one",
        )
        if subsets != ["all"]:
            meta = _load_matched_metadata(
                "all",
                p1_value_subset=subsets,
                p1_value_question_col="source_question_unpadded",
            )[["original_id", "instance"]].drop_duplicates()
            merged = merged.merge(meta, on=["original_id", "instance"], validate="one_to_one")
        out = pd.DataFrame({
            "original_id": merged["original_id"].astype(int),
            "instance":    merged["instance"].astype(int),
            "source_question": merged["question_source"].astype(str),
            "target_question": merged["question_target"].astype(str),
            "source_answer":   merged["answer_source"].astype(str),
            "target_answer":   merged["answer_target"].astype(str),
            "len_delta_after": 0,
            "source_label":    "symbolic",
            "target_label":    "p1",
            "source_cot_prefix": merged["original_cot"].map(cot_prefix_before_hash),
        })
        out = _filter_distinct_answers(out, source_col="source_answer", target_col="target_answer")
    elif contrast == "p1_vs_padded_symbolic":
        if padded_dataset not in PADDED_TFM_DATASETS:
            raise ValueError(
                "p1_vs_padded_symbolic needs a concrete padded transformer dataset; "
                f"use --padded_dataset {sorted(PADDED_TFM_DATASETS)}."
            )
        entry = PADDED_TFM_DATASETS[padded_dataset]
        if padded_dataset == "legacy_symbolic_padded":
            if length_filter not in entry:
                raise ValueError(
                    f"--padded_dataset legacy_symbolic_padded requires "
                    f"--length_filter in {sorted(entry)} (got {length_filter}). "
                    "Legacy inference CSVs are pre-filtered per length subset; "
                    "there is no `all` legacy file."
                )
            padded_subdir, padded_name = entry[length_filter]
        else:
            padded_subdir, padded_name = entry
        source = _load_split_results(padded_subdir, padded_name, model_name)
        p1 = _load_split_results("gsm_p1", "gsm_p1", model_name)

        source[correctness_col] = _coerce_bool(source[correctness_col])
        p1[correctness_col] = _coerce_bool(p1[correctness_col])
        source = source[source[correctness_col] == True].copy()
        p1 = p1[p1[correctness_col] == True].copy()

        if padded_dataset == "gsm_padded_symbolic":
            # Per-instance length-matched at construction, but NOT entity-
            # aligned to gsm_p1 — inherits sym's bindings, so joining on
            # (oid, inst) yields semantically mismatched pairs. Disallow this
            # combination explicitly to avoid silent meaningless patching runs.
            raise ValueError(
                "--padded_dataset gsm_padded_symbolic is NOT entity-aligned "
                "to gsm_p1 (it inherits gsm_symbolic's bindings). Use "
                "gsm_padded_symbolic_p1_aligned for p1_vs_padded_symbolic "
                "patching."
            )
        elif padded_dataset == "gsm_padded_symbolic_p1_aligned":
            # Entity-aligned to gsm_p1 by construction (built from matched-
            # pairs CSV's source_question_unpadded). Construction targets
            # per-instance length match, but a small fraction of rows are
            # `unresolved` (couldn't hit p1's exact token count); compute
            # len_delta_after at load so --length_filter can drop them.
            # p1_value_subset filtering would need p1_introduced_numbers
            # from matched-pairs; support `all` for now and add subset
            # filtering later if needed.
            if subsets != ["all"]:
                raise ValueError(
                    "--p1_value_subset != 'all' is not yet supported with "
                    "--padded_dataset gsm_padded_symbolic_p1_aligned."
                )
            # len_delta computed after the source/p1 merge below.
        elif padded_dataset == "legacy_symbolic_padded":
            # Legacy pre-filtered inference: len_delta is 0 (delta0) or
            # already constrained to |delta|<2 (lt2) at construction time.
            # The matched-pairs CSV stores the exact deltas; we'll merge
            # them in for downstream consumers that read len_delta_after.
            meta = _load_matched_metadata(length_filter, p1_value_subset=subsets)[
                ["original_id", "instance", "len_delta_after"]
            ].drop_duplicates()
            source = source.merge(meta, on=["original_id", "instance"], validate="one_to_one")

        # Include each side's own `original_cot` so we can compute baseline
        # gold logprobs with each question's OWN CoT (not the shared donor).
        # padded.original_cot → original_cot_source (after the suffix merge);
        # p1.original_cot → original_cot_target.
        merged = source.merge(
            p1[
                [
                    "original_id",
                    "instance",
                    "question",
                    "answer",
                    "original_cot",
                    correctness_col,
                ]
            ],
            on=["original_id", "instance"],
            suffixes=("_source", "_target"),
            validate="one_to_one",
        )
        merged = _filter_distinct_answers(
            merged, source_col="answer_source", target_col="answer_target"
        )
        if "len_delta_after" not in merged.columns:
            # New datasets don't carry len_delta_after — compute it from the
            # tokenized question lengths after merging on (oid, inst).
            merged["len_delta_after"] = _compute_len_delta_column(
                merged["question_source"].astype(str).tolist(),
                merged["question_target"].astype(str).tolist(),
                model_name,
            )
        merged = _filter_subset(merged, length_filter)

        # Donor CoT must come from the original (un-padded) gsm_symbolic, NOT
        # from padded_symbolic's own generated CoT and NOT from p1. The padded
        # side regenerates inference at its longer length and produces a
        # different (sometimes worse) CoT; we want the clean symbolic-aligned
        # reasoning as the donor, mirroring `--cot_source noop_clean` for
        # cot_swap. Join on (oid, inst) with sym's cot-correct rows; drop pairs
        # lacking a matching clean symbolic CoT.
        sym = _load_split_results("gsm_symbolic", "gsm_symbolic_to_p1", model_name)
        sym[correctness_col] = _coerce_bool(sym[correctness_col])
        sym = sym[sym[correctness_col] == True].copy()
        sym_cot = sym[["original_id", "instance", "original_cot"]].rename(
            columns={"original_cot": "_symbolic_cot"}
        ).astype({"original_id": int, "instance": int})
        before = len(merged)
        merged = merged.merge(
            sym_cot,
            on=["original_id", "instance"],
            how="inner",
            validate="one_to_one",
        )
        dropped = before - len(merged)
        if dropped:
            print(
                f"_load_p1_contrast_data: dropped {dropped}/{before} rows "
                f"lacking a cot-correct gsm_symbolic CoT for the donor."
            )

        out = pd.DataFrame(
            {
                "original_id": merged["original_id"].astype(int),
                "instance": merged["instance"].astype(int),
                "source_question": merged["question_source"].astype(str),
                "target_question": merged["question_target"].astype(str),
                "source_answer": merged["answer_source"].astype(str),
                "target_answer": merged["answer_target"].astype(str),
                "len_delta_after": merged["len_delta_after"].astype(int),
                "source_label": "padded_symbolic",
                "target_label": "p1",
                # Donor CoT (symbolic-aligned). Used for the PATCHING prompts
                # and for the shared-donor-CoT baseline (logits-of-####).
                "source_cot_prefix": merged["_symbolic_cot"].map(cot_prefix_before_hash),
                # Each side's OWN CoT (model's natural generation on that side's
                # question). Used for the gold-logprob baseline measurement so
                # source_minus_target_gold_logprob reflects the natural cliff,
                # not a shared-CoT artefact.
                "source_own_cot_prefix": merged["original_cot_source"].map(cot_prefix_before_hash),
                "target_own_cot_prefix": merged["original_cot_target"].map(cot_prefix_before_hash),
            }
        )
    else:
        raise ValueError(f"Unknown P1 contrast: {contrast}")

    out = out.dropna(subset=["source_question", "target_question", "source_cot_prefix"])
    return out.sort_values(["original_id", "instance"]).reset_index(drop=True)


def run_p1_readiness_logit_lens(
    model,
    tokenizer,
    model_name,
    contrast,
    padded_dataset="gsm_padded_symbolic_p1_aligned",
    p1_value_subset="all",
    length_filter="all",
    max_examples=None,
    max_pairs_per_template=None,
    seed=0,
    overwrite=False,
    verbose=False,
):
    subsets = _normalize_subsets(p1_value_subset)
    if subsets != ["all"] and contrast not in {"p1_vs_symbolic", "p1_vs_padded_symbolic"}:
        raise ValueError(
            f"--p1_value_subset is only supported with P1 contrasts "
            f"(got contrast={contrast})"
        )
    df = _load_p1_contrast_data(
        contrast, padded_dataset, model_name, p1_value_subset=subsets,
        max_pairs_per_template=max_pairs_per_template, seed=seed,
        length_filter=length_filter,
    )
    if max_examples is not None:
        df = df.head(max_examples).copy()

    print(
        f"Loaded {len(df)} {contrast} examples "
        f"({df['original_id'].nunique()} templates, padded_dataset={padded_dataset}, "
        f"length_filter={length_filter}, p1_value_subset={'+'.join(subsets)})"
    )
    subset_dir = "all" if subsets == ["all"] else "+".join(subsets)
    out_path = OUT_DIR / model_name / contrast / padded_dataset / length_filter / subset_dir / "readiness_logit_lens.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink()

    tok_hash = tokenizer.encode("####", add_special_tokens=False)
    if len(tok_hash) != 1:
        raise ValueError(f"Expected single-token ####, got token ids {tok_hash}")
    tok_hash = tok_hash[0]

    skip_ctr = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prefix_text = row["source_cot_prefix"]
        if not isinstance(prefix_text, str) or not prefix_text:
            skip_ctr += 1
            if verbose:
                print(f"Skipping original_id={row['original_id']} instance={row['instance']}: empty CoT prefix")
            continue

        source_ready_inputs = build_chat_inputs(
            row["source_question"], model, tokenizer, answer_prefix=prefix_text
        )
        target_ready_inputs = build_chat_inputs(
            row["target_question"], model, tokenizer, answer_prefix=prefix_text
        )

        source_logps = get_token_logprob_lens(
            forward_with_hooks(model, source_ready_inputs), model, tok_hash
        )
        target_logps = get_token_logprob_lens(
            forward_with_hooks(model, target_ready_inputs), model, tok_hash
        )
        readiness_gap = source_logps - target_logps

        result = {
            "original_id": int(row["original_id"]),
            "instance": int(row["instance"]),
            "contrast": contrast,
            "padded_dataset": padded_dataset,
            "length_filter": length_filter,
            "source_label": row["source_label"],
            "target_label": row["target_label"],
            "source_answer": row["source_answer"],
            "target_answer": row["target_answer"],
            "len_delta_after": int(row["len_delta_after"]),
            "hash_token_id": int(tok_hash),
            "source_question": row["source_question"],
            "target_question": row["target_question"],
            "symbolic_cot_prefix": prefix_text,
            "readout_name": "log P(next token = ####)",
            "source_hash_logprobs": to_serializable(source_logps),
            "target_hash_logprobs": to_serializable(target_logps),
            "source_readout_values": to_serializable(source_logps),
            "target_readout_values": to_serializable(target_logps),
            "readiness_gaps": to_serializable(readiness_gap),
            "readout_deltas": to_serializable(readiness_gap),
        }
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    if verbose:
        print(f"Skipped {skip_ctr} rows with missing generated ####")
    print(f"Results → {out_path}")
    return out_path


def _noop_swap_json_path(model_name, drop_swap_correct_filter, source="noop"):
    subset_dir = "all" if drop_swap_correct_filter else "swap_failed"
    contrast_dir = {
        "noop":                 "noop_vs_symbolic",
        "noop_clean":           "noop_clean_vs_symbolic",
        "filler_vs_noop_clean": "noop_clean_vs_filler",
    }[source]
    return (
        OUT_DIR / model_name / contrast_dir / subset_dir
        / "cot_swap_logit_lens.json"
    )


def run_cot_swap_logit_lens(model, tokenizer, model_name, verbose=False,
                             drop_swap_correct_filter=False, overwrite=False,
                             noop_source="noop"):
    df = load_cot_swap_data(
        drop_swap_correct_filter=drop_swap_correct_filter,
        source=noop_source,
    )
    label = "all sym-correct" if drop_swap_correct_filter else "cot-swap-wrong"
    print(f"Loaded {len(df)} {label} examples (source={noop_source})")

    out_path = _noop_swap_json_path(model_name, drop_swap_correct_filter, noop_source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink()

    tok_hash  = tokenizer.encode("####", add_special_tokens=False)[0]  # 827
    tok_space = tokenizer.encode(" ",    add_special_tokens=False)[0]  # 220

    skip_ctr = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sym_prompt  = row["sym_question"]
        noop_prompt = row["noop_question"]
        reasoning   = row["sym_reasoning"]

        if pd.isna(sym_prompt) or pd.isna(noop_prompt) or pd.isna(reasoning):
            skip_ctr += 1
            continue

        reasoning = str(reasoning)

        sym_inputs,  _ = tokenize(sym_prompt,  model, tokenizer,
                                  instruction=COT_INSTRUCTION, answer_prefix=reasoning)
        noop_inputs, _ = tokenize(noop_prompt, model, tokenizer,
                                  instruction=COT_INSTRUCTION, answer_prefix=reasoning)

        # Extend each run to its natural "#### " position
        sym_extended,  _ = extend_to_answer_position(
            model, tokenizer, sym_inputs,  reasoning, tok_hash, tok_space)
        noop_extended, _ = extend_to_answer_position(
            model, tokenizer, noop_inputs, reasoning, tok_hash, tok_space)

        if sym_extended is None or noop_extended is None:
            skip_ctr += 1
            if verbose:
                print(f"Skipping: '####' not found in generation")
            continue

        # Target: sym answer integer token
        sym_ans    = int(row["sym_answer"])
        target_id  = tokenizer.encode(str(sym_ans), add_special_tokens=False)[0]

        sym_logprobs  = get_token_logprob_lens(forward_with_hooks(model, sym_extended),  model, target_id)
        noop_logprobs = get_token_logprob_lens(forward_with_hooks(model, noop_extended), model, target_id)

        sc = row["swap_correct"]
        sa = row["swap_answer"]
        result = {
            "id":           int(row["id"]),
            "instance":     int(row["instance"]),
            "split":        int(row["split"]),
            "sym_answer":   sym_ans,
            "swap_answer":  None if pd.isna(sa) else sa,
            "swap_correct": None if pd.isna(sc) else bool(sc),
            "sym_logprobs":  to_serializable(sym_logprobs),
            "noop_logprobs": to_serializable(noop_logprobs),
        }

        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    if verbose:
        print(f"Skipped {skip_ctr} rows with missing data")
    print(f"Results → {out_path}")
    return out_path


def _template_means(values, cluster_ids):
    """Average rows within each template/cluster before cross-template summaries."""
    values = np.asarray(values, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    unique = np.unique(cluster_ids)
    means = np.stack([values[cluster_ids == c].mean(axis=0) for c in unique])
    return unique, means


def _bootstrap_ci(values, n_boot=1000, ci=0.95, seed=0):
    """Bootstrap CI for the per-layer mean over independent rows."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.shape[0] == 1:
        return values[0].copy(), values[0].copy()
    boot = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        rows = rng.integers(0, values.shape[0], size=values.shape[0])
        boot[b] = values[rows].mean(axis=0)
    alpha = (1 - ci) / 2
    return np.quantile(boot, alpha, axis=0), np.quantile(boot, 1 - alpha, axis=0)


STAGE_BOUNDARIES = [
    (36, "Symbolic\nformulation"),
]


def _style_layer_axis(ax, num_layers, show_xlabel=True, stage_boundaries=None):
    """Apply the shared stage-boundary + x-tick style from template_similarity.

    Lens default markers only L36 (formulation onset); the L80 readout marker
    is intentionally omitted because the lens divergence already terminates
    visibly at the axis end. Callers can pass their own ``stage_boundaries``
    (e.g. the patching plot adds L80 back).
    """
    if stage_boundaries is None:
        stage_boundaries = STAGE_BOUNDARIES
    stage_ticks = []
    for boundary, label in stage_boundaries:
        ax.axvline(boundary, color="#333333", lw=0.9, alpha=0.55, ls="--")
        ax.text(
            boundary, 0.95, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", rotation=90, fontsize=7.5, color="#333333",
        )
        stage_ticks.append(boundary)

    interval = 20
    default_ticks = list(range(0, num_layers, interval))
    ticks = sorted(set(default_ticks + stage_ticks))
    cleaned = [t for t in ticks
               if t in stage_ticks
               or all(abs(t - s) >= 4 for s in stage_ticks)]
    ax.set_xticks(cleaned)
    for tlabel in ax.get_xticklabels():
        try:
            v = int(tlabel.get_text())
        except (TypeError, ValueError):
            continue
        if v in stage_ticks:
            tlabel.set_fontweight("bold")
            tlabel.set_color("#333333")

    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    if show_xlabel:
        ax.set_xlabel("Layer")


def _save_per_template_supp(layers, div_template, divergence, n, n_clusters,
                             out_path):
    """Per-template spaghetti panel, saved as a separate supplementary figure."""
    num_layers = len(layers)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i in range(div_template.shape[0]):
        ax.plot(layers, div_template[i], color="#777777", alpha=0.28, lw=0.8)
    ax.plot(layers, divergence, color="#333333", lw=1.8, label="Template mean")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_ylabel("Log-prob difference")
    _style_layer_axis(ax, num_layers)
    ax.legend(fontsize=9.5, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-template supp plot → {out_path}")


def plot_cot_swap_logit_lens(model_name, n_boot=1000, ci=0.95,
                              drop_swap_correct_filter=False, noop_source="noop"):
    results_path = _noop_swap_json_path(model_name, drop_swap_correct_filter, noop_source)
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    sym  = np.array([r["sym_logprobs"]  for r in results])
    noop = np.array([r["noop_logprobs"] for r in results])
    if sym.ndim == 3:
        sym  = sym.squeeze(-1)
        noop = noop.squeeze(-1)
    cluster_ids = np.array([r["id"] for r in results])

    valid = np.isfinite(sym).all(axis=1) & np.isfinite(noop).all(axis=1)
    sym, noop, cluster_ids = sym[valid], noop[valid], cluster_ids[valid]
    n, num_layers = sym.shape
    layers = np.arange(1, num_layers + 1)  # post-layer convention: lens index i = residual after block i = template_similarity L(i+1)

    template_ids, sym_template = _template_means(sym, cluster_ids)
    _, noop_template = _template_means(noop, cluster_ids)
    div_template = sym_template - noop_template

    sym_mean = sym_template.mean(axis=0)
    noop_mean = noop_template.mean(axis=0)
    divergence = div_template.mean(axis=0)

    sym_lo, sym_hi = _bootstrap_ci(sym_template, n_boot, ci)
    noop_lo, noop_hi = _bootstrap_ci(noop_template, n_boot, ci)
    div_lo, div_hi = _bootstrap_ci(div_template, n_boot, ci)
    n_clusters = len(template_ids)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    colors = {"sym": "#2166ac", "noop": "#d6604d"}

    ax0.plot(layers, sym_mean,  color=colors["sym"],  lw=1.8, label="Sym question (clean)")
    ax0.fill_between(layers, sym_lo, sym_hi, color=colors["sym"], alpha=0.18)
    ax0.plot(layers, noop_mean, color=colors["noop"], lw=1.8, label="NoOp question (corrupted)")
    ax0.fill_between(layers, noop_lo, noop_hi, color=colors["noop"], alpha=0.18)
    ax0.set_ylabel("Mean log P(answer)")
    _style_layer_axis(ax0, num_layers, show_xlabel=False)
    ax0.legend(fontsize=9.5, loc="lower left", frameon=True)

    ax1.plot(layers, divergence, color="#333333", lw=1.8, label="Sym − NoOp")
    ax1.fill_between(layers, div_lo, div_hi, color="#333333", alpha=0.2)
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.set_ylabel("Sym − NoOp (log P)")
    _style_layer_axis(ax1, num_layers)
    ax1.legend(fontsize=9.5, loc="upper left", frameon=True)

    plt.tight_layout()
    save_path = results_path.with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {save_path}")

    supp_path = save_path.with_name(save_path.stem + "_per_template.png")
    _save_per_template_supp(layers, div_template, divergence, n, n_clusters, supp_path)


def plot_p1_readiness_logit_lens(model_name, contrast, padded_dataset="gsm_padded_symbolic_p1_aligned",
                                 p1_value_subset="all", length_filter="all",
                                 n_boot=1000, ci=0.95):
    subsets = _normalize_subsets(p1_value_subset)
    subset_dir = "all" if subsets == ["all"] else "+".join(subsets)
    results_path = OUT_DIR / model_name / contrast / padded_dataset / length_filter / subset_dir / "readiness_logit_lens.jsonl"
    results = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    source = np.array([r["source_hash_logprobs"] for r in results])
    target = np.array([r["target_hash_logprobs"] for r in results])
    cluster_ids = np.array([r["original_id"] for r in results])
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target, cluster_ids = source[valid], target[valid], cluster_ids[valid]
    n, num_layers = source.shape
    layers = np.arange(1, num_layers + 1)  # post-layer convention: lens index i = residual after block i = template_similarity L(i+1)

    template_ids, source_template = _template_means(source, cluster_ids)
    _, target_template = _template_means(target, cluster_ids)
    div_template = source_template - target_template

    source_mean = source_template.mean(axis=0)
    target_mean = target_template.mean(axis=0)
    divergence = div_template.mean(axis=0)

    src_lo, src_hi = _bootstrap_ci(source_template, n_boot, ci)
    tgt_lo, tgt_hi = _bootstrap_ci(target_template, n_boot, ci)
    div_lo, div_hi = _bootstrap_ci(div_template, n_boot, ci)
    n_clusters = len(template_ids)

    source_label = results[0].get("source_label", "source") if results else "source"
    target_label = results[0].get("target_label", "target") if results else "target"

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)
    colors = {"source": "#2166ac", "target": "#d6604d"}

    ax0.plot(layers, source_mean, color=colors["source"], lw=1.8, label=source_label)
    ax0.fill_between(layers, src_lo, src_hi, color=colors["source"], alpha=0.18)
    ax0.plot(layers, target_mean, color=colors["target"], lw=1.8, label=target_label)
    ax0.fill_between(layers, tgt_lo, tgt_hi, color=colors["target"], alpha=0.18)
    ax0.set_ylabel("Mean log P(answer marker)")
    _style_layer_axis(ax0, num_layers, show_xlabel=False)
    ax0.legend(fontsize=9.5, loc="lower left", frameon=True)

    ax1.plot(layers, divergence, color="#333333", lw=1.8,
             label=f"{source_label} − {target_label}")
    ax1.fill_between(layers, div_lo, div_hi, color="#333333", alpha=0.2)
    ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax1.set_ylabel(f"{source_label} − {target_label} (log P)")
    _style_layer_axis(ax1, num_layers)
    ax1.legend(fontsize=9.5, loc="upper left", frameon=True)

    plt.tight_layout()
    save_path = results_path.with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot → {save_path}")

    supp_path = save_path.with_name("readiness_logit_lens_per_template.png")
    _save_per_template_supp(layers, div_template, divergence, n, n_clusters, supp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
        choices=list(MODEL_NAME_MAP.keys()),
    )
    parser.add_argument(
        "--contrast",
        choices=["noop_swap", "p1_vs_symbolic", "p1_vs_padded_symbolic"],
        default="noop_swap",
        help="Which CoT-swap logit lens contrast to run.",
    )
    parser.add_argument(
        "--padded_dataset",
        choices=[
            "all",
            "gsm_padded_symbolic",
            "gsm_padded_symbolic_p1_aligned",
            "legacy_symbolic_padded",
        ],
        default="gsm_padded_symbolic_p1_aligned",
        help=(
            "Which padded sym dataset for P1 contrasts. Use `all` for "
            "p1_vs_symbolic (no padding). For p1_vs_padded_symbolic use "
            "gsm_padded_symbolic_p1_aligned (entity-aligned to p1) — the "
            "loader rejects gsm_padded_symbolic for this contrast. "
            "legacy_symbolic_padded loads the pre-filtered legacy CSVs; "
            "--length_filter selects delta0 or lt2."
        ),
    )
    parser.add_argument(
        "--length_filter",
        choices=list(LENGTH_FILTERS),
        default="all",
        help=(
            "Length-delta filter applied to p1_vs_padded_symbolic: "
            "delta0 → tok_len(source)==tok_len(p1); lt2 → |Δ|<2; all → no filter. "
            "For legacy_symbolic_padded, selects which pre-filtered legacy file "
            "to load (delta0 or lt2; `all` is not available)."
        ),
    )
    parser.add_argument(
        "--p1_value_subset",
        nargs="+",
        choices=["all", "no_value", "with_value", "value_present", "value_absent"],
        default=["all"],
        help=(
            "For P1 contrasts. One or more subsets to run sequentially "
            "(each produces its own jsonl/png). Filter pairs by whether the P1 "
            "transformation introduced a numeric/fractional value "
            "(`p1_introduced_numbers`) and whether that value appears in the symbolic "
            "source question. For p1_vs_padded_symbolic this is the padded source; "
            "for p1_vs_symbolic this is the unpadded source. "
            "no_value: P1 introduced no value (only wording). "
            "with_value: P1 introduced ≥1 value (any presence). "
            "value_present: source contains all P1-introduced values. "
            "value_absent: source contains none of the P1-introduced values."
        ),
    )
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument(
        "--max_pairs_per_template", type=int, default=None,
        help="For p1_vs_symbolic only: cap random 1-to-1 sym/p1 pairs per template.",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for p1_vs_symbolic instance pairing.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument(
        "--include_recovered",
        action="store_true",
        help=(
            "noop_swap only: drop the `~swap_correct` filter and keep all rows "
            "where the natural sym CoT was correct (recovered + failed swaps). "
            "Outputs land in <contrast_dir>/all/; the default (swap-failed only) "
            "lands in <contrast_dir>/swap_failed/."
        ),
    )
    parser.add_argument(
        "--noop_source",
        choices=["noop", "noop_clean", "filler_vs_noop_clean"],
        default="noop",
        help=(
            "noop_swap only: which noop variant to read from. "
            "`noop` = the legacy vLLM + gsm_noop CSV (results/.../cot_swap/). "
            "`noop_clean` = the transformers + gsm_noop_clean rebuild "
            "(results/.../cot_swap_noop_clean/). `filler_vs_noop_clean` = the "
            "filler-correct ∩ noop_clean-wrong rebuild that swaps in the filler "
            "CoT trace; `sym_*` columns hold the filler side. For the rebuild "
            "sources, run disentangled_evaluation/cot_swap_experiment.py first "
            "to populate `swap_correct` before dropping --include_recovered."
        ),
    )
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]

    subsets = _normalize_subsets(args.p1_value_subset)

    if args.include_recovered and args.contrast != "noop_swap":
        parser.error("--include_recovered only applies to --contrast noop_swap")
    if args.noop_source != "noop" and args.contrast != "noop_swap":
        parser.error("--noop_source only applies to --contrast noop_swap")

    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        if args.contrast == "noop_swap":
            run_cot_swap_logit_lens(
                model, tokenizer, model_name,
                verbose=args.verbose,
                drop_swap_correct_filter=args.include_recovered,
                overwrite=args.overwrite,
                noop_source=args.noop_source,
            )
        else:
            run_p1_readiness_logit_lens(
                model,
                tokenizer,
                model_name,
                contrast=args.contrast,
                padded_dataset=args.padded_dataset,
                p1_value_subset=subsets,
                length_filter=args.length_filter,
                max_examples=args.max_examples,
                max_pairs_per_template=args.max_pairs_per_template,
                seed=args.seed,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )

    if args.contrast == "noop_swap":
        plot_cot_swap_logit_lens(
            model_name,
            drop_swap_correct_filter=args.include_recovered,
            noop_source=args.noop_source,
        )
    else:
        plot_p1_readiness_logit_lens(
            model_name, args.contrast, args.padded_dataset, subsets,
            length_filter=args.length_filter,
        )
