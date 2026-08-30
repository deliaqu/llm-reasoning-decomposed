"""
Cross-instance / cross-template residual similarity experiment.

Tests whether the four-stage decomposition of word-problem reasoning is
mechanically real in residual stream representations.

Hypothesis: Early-to-mid layers encode template-level structure (Stages 1–2:
operation type, entity schema, formula skeleton) and are therefore similar
across instances of the same problem template.  Later layers encode
instance-level content (Stages 3–4: concrete operand values, numeric answer)
and diverge within-template.

Measured by: mean pairwise cosine similarity within-template vs cross-template
at each layer.  The layer at which the within/cross gap closes is the
Stage 2 → Stage 3 boundary.

Within-template pairs:   same original_id, different instance (same formula
                         skeleton, different concrete numbers)
Cross-template pairs:    different original_id (different formula skeleton)

Usage:
    python run_template_similarity.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_template_similarity.py --model_id meta-llama/Llama-3.3-70B-Instruct --correct_only
    python run_template_similarity.py --model_id meta-llama/Llama-3.3-70B-Instruct --mode unspecified
    python run_template_similarity.py --plot_only
    python run_template_similarity.py --plot_only --model_id meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import glob
import json
import os as _os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
try:
    import torch
except ImportError:  # Plot-only mode does not need torch.
    torch = None
try:
    from tqdm import tqdm
except ImportError:  # Plot-only mode does not need progress bars.
    tqdm = None

from config import (
    ANSWER_PREFIX, COT_INSTRUCTION, DIRECT_INSTRUCTION,
    LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP,
)

ANSWER_MARKER = "####"

# Plots and template_similarity result JSONs live in the project results tree.
OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "template_similarity"
RESULT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent.parent / "results" / "disentangled_evaluation"
# pair_metadata.csv emitted by build_pair_inference_data.py --dedup_inference
# carries the full pair structure (pair_id, label, value, role) keyed by the
# `question` column. load_gsm_dataset joins it back when the inference CSVs
# came from the dedup flow and therefore lack inline pair columns.
PAIR_METADATA_CSV = (
    Path(LOGIT_LENS_RESULT_DIR).parent.parent
    / "data" / "gsm_symbolic" / "split_datasets_pairs" / "pair_metadata.csv"
)

# Hidden-state caches (.npy + meta CSV) are large
# (~6 GB per mode for gsm_symbolic, ~100 GB per mode for gsm_symbolic_pairs)
# and don't belong in the project results tree. Override with $CACHE_DIR.
CACHE_DIR = Path(_os.environ.get("CACHE_DIR", str(Path(LOGIT_LENS_RESULT_DIR).parent.parent / "cache"))) / "template_similarity"

# Per-dataset overrides for the hidden-state cache root. Datasets not listed
# here use CACHE_DIR. Add entries here if a specific dataset needs a different
# cache location.
CACHE_ROOT_OVERRIDES = {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATASET_NAMES = {
    "gsm_symbolic": "GSM-Symbolic",
    "gsm_p1": "GSM-P1",
    "gsm_p2": "GSM-P2",
    "gsm_symbolic_pairs": "GSM-Symbolic-Pairs",
    "gsm_symbolic_padded_to_p1_len_delta_0":
        "GSM-Symbolic-Padded-Δ0",
    "gsm_symbolic_padded_to_p1_len_delta_abs_lt2":
        "GSM-Symbolic-Padded-|Δ|<2",
    "gsm_noop_clean": "GSM-NoOp-Clean",
    "gsm_noop_clean_hard": "GSM-NoOp-Clean-Hard",
    "gsm_filler": "GSM-Filler",
    "gsm_filler_df": "GSM-Filler-DF",
}

# Datasets whose inference outputs live under the transformers_direct subdir
# and use the `_transformers` model-name suffix.
TRANSFORMERS_DIRECT_DATASETS = {
    "gsm_symbolic_pairs",
    "gsm_symbolic_padded_to_p1_len_delta_0",
    "gsm_symbolic_padded_to_p1_len_delta_abs_lt2",
    "gsm_filler",
    "gsm_filler_df",
}
PAIRED_ALIGN_KEYS = [
    "original_id", "instance", "pair_id", "label", "value",
]


def _correctness_col(mode: str) -> str:
    # CoT-framed modes filter on CoT correctness so they share a prompt set
    # with `cot` mode (same data, different readout position).
    if mode in ("cot", "cot_pre_reasoning"):
        return "original_cot_correctness"
    # Eval CSVs only record direct and CoT correctness. For the no-instruction
    # prompt variants, reuse direct correctness as the closest available
    # no-CoT supervision signal.
    return "original_direct_correctness"


def load_gsm_dataset(
    model_id: str,
    dataset_name: str = "gsm_symbolic",
    correct_only: bool = False,
    mode: str = "direct",
) -> pd.DataFrame:
    # Eval result files use the raw HuggingFace model name suffix (case-preserved),
    # e.g. "Llama-3.3-70B-Instruct", not the lowercased interpretability short form.
    if dataset_name not in DATASET_NAMES:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Choices: {', '.join(DATASET_NAMES)}"
        )
    base_suffix = model_id.split("/")[-1]
    # Padded-symbolic variants share the gsm_symbolic/ subdir; pairs and
    # everything else use a per-dataset subdir.
    if dataset_name.startswith("gsm_symbolic_padded"):
        tfm_subdir = "gsm_symbolic"
    else:
        tfm_subdir = dataset_name
    tfm_pattern = str(
        RESULT_DIR / "transformers_direct" / tfm_subdir /
        f"{dataset_name}_set_*_results_{base_suffix}_transformers.csv"
    )
    vllm_pattern = str(
        RESULT_DIR / dataset_name /
        f"{dataset_name}_set_*_results_{base_suffix}.csv"
    )
    # For datasets historically inferred via transformers_direct (pairs,
    # padded variants), check that path first; otherwise prefer vLLM and
    # fall back to transformers_direct so models inferenced exclusively
    # via transformers_direct (the new generalization set) still resolve.
    if dataset_name in TRANSFORMERS_DIRECT_DATASETS:
        primary_pattern, fallback_pattern = tfm_pattern, vllm_pattern
    else:
        primary_pattern, fallback_pattern = vllm_pattern, tfm_pattern
    files = sorted(glob.glob(primary_pattern))
    if not files:
        files = sorted(glob.glob(fallback_pattern))
    if not files:
        raise FileNotFoundError(
            f"No {DATASET_NAMES[dataset_name]} files for {base_suffix} matching:\n"
            f"  primary:  {primary_pattern}\n"
            f"  fallback: {fallback_pattern}"
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if dataset_name == "gsm_symbolic_pairs" and "pair_id" not in df.columns:
        # `build_pair_inference_data.py --dedup_inference` writes inference
        # CSVs without inline pair structure (one row per unique question,
        # ~37% smaller). Join pair_metadata.csv on `question` so downstream
        # paired-CV code (cache writer + entity/op probes) sees the same
        # schema as the non-dedup flow.
        if not PAIR_METADATA_CSV.exists():
            raise FileNotFoundError(
                "Paired inference CSVs lack `pair_id` (dedup-inference mode), "
                f"but {PAIR_METADATA_CSV} is missing. Rerun "
                "`build_pair_inference_data.py` (with or without "
                "`--dedup_inference`) so pair metadata is emitted alongside."
            )
        pair_meta = pd.read_csv(PAIR_METADATA_CSV)
        inf_cols = [c for c in df.columns
                    if c.startswith("original_direct")
                    or c.startswith("original_cot")]
        if not inf_cols:
            raise ValueError(
                "Dedup-mode inference CSVs have no `original_direct*`/"
                "`original_cot*` output columns to join with pair_metadata.csv."
            )
        if "question" not in df.columns or "question" not in pair_meta.columns:
            raise ValueError(
                "Auto-join needs a `question` column in both inference CSVs "
                "and pair_metadata.csv to align dedup rows with pair halves."
            )
        before = len(df)
        inf_unique = (df[["question"] + inf_cols]
                      .drop_duplicates(subset=["question"], keep="first"))
        joined = pair_meta.merge(inf_unique, on="question", how="left")
        n_unmatched = joined[inf_cols[0]].isna().sum()
        if n_unmatched:
            print(f"  warning: {n_unmatched}/{len(joined)} pair rows had no "
                  "matching inference output; dropping (likely a partial "
                  "inference run).")
            joined = joined.dropna(subset=[inf_cols[0]]).reset_index(drop=True)
        print(f"  auto-joined dedup-mode paired inference: {before} unique "
              f"question-rows -> {len(joined)} pair rows "
              f"({pair_meta['pair_id'].nunique()} pairs)")
        df = joined
    if mode == "cot" and "original_cot" not in df.columns:
        raise ValueError(
            "CoT mode requires the `original_cot` column in eval CSVs, "
            "but it is missing."
        )
    if correct_only:
        before = len(df)
        col = _correctness_col(mode)
        correctness = _coerce_bool(df[col])
        df = df[correctness == True].reset_index(drop=True)
        print(f"Filtered to correct instances ({col}): {len(df)}/{before}")
    return df


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Parse CSV boolean columns without treating non-empty strings as True."""
    normalized = series.map(
        lambda x: x.strip().lower() if isinstance(x, str) else x
    )
    return normalized.map({
        True: True, False: False,
        1: True, 0: False, 1.0: True, 0.0: False,
        "true": True, "false": False,
        "1": True, "0": False,
    }).astype("boolean")


# ---------------------------------------------------------------------------
# Hidden state extraction
# ---------------------------------------------------------------------------

def _fmt_direct(q: str, tokenizer) -> str:
    msgs = [{"role": "user", "content": f"{q}\n{DIRECT_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    ) + ANSWER_PREFIX


def _fmt_unspecified(q: str, tokenizer) -> str:
    msgs = [{"role": "user", "content": q}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _fmt_cot_pre_reasoning(q: str, tokenizer) -> str:
    """Cot-framed prompt with the CoT instruction in place but no cached
    reasoning. Residual is read at the final token (assistant-header newline),
    i.e. one token before the model would generate its first reasoning token."""
    msgs = [{"role": "user", "content": f"{q}\n{COT_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _question_token_span(q: str, tokenizer) -> tuple[int, int]:
    """Return the un-padded [start, stop) token span of the question within the
    chat-formatted prompt produced by `_fmt_unspecified`.

    Works by comparing the full prompt's token ids against the empty-content
    prompt's token ids and finding the diverging region (= the question).
    Robust to any chat template that renders user content by substitution.
    """
    full = _fmt_unspecified(q, tokenizer)
    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False, add_generation_prompt=True,
    )
    full_ids = tokenizer(full, add_special_tokens=True)["input_ids"]
    empty_ids = tokenizer(empty, add_special_tokens=True)["input_ids"]
    # Longest common prefix.
    common = 0
    while (common < len(full_ids) and common < len(empty_ids)
           and full_ids[common] == empty_ids[common]):
        common += 1
    # Longest common suffix (after the diverging region).
    suffix = 0
    while (suffix < len(full_ids) - common
           and suffix < len(empty_ids) - common
           and full_ids[-1 - suffix] == empty_ids[-1 - suffix]):
        suffix += 1
    # Question tokens occupy [common, len(full_ids) - suffix).
    return common, len(full_ids) - suffix


def _last_question_token_index(q: str, tokenizer) -> int:
    """Return the un-padded token index of the last question token."""
    _start, stop = _question_token_span(q, tokenizer)
    return stop - 1


def _fmt_cot(q: str, cot_response: str, tokenizer):
    """CoT prompt: question + COT_INSTRUCTION + cached reasoning ending at '#### '.
    Returns None if the cached response lacks the answer marker."""
    if not isinstance(cot_response, str) or ANSWER_MARKER not in cot_response:
        return None
    reasoning_prefix = cot_response.rsplit(ANSWER_MARKER, 1)[0].rstrip()
    msgs = [{"role": "user", "content": f"{q}\n{COT_INSTRUCTION}"}]
    assistant_open = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    return f"{assistant_open}{reasoning_prefix}\n\n{ANSWER_PREFIX}"


def extract_hidden_states(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int = 4,
    mode: str = "direct",
    max_length: int = 4096,
) -> tuple[np.ndarray, list[int]]:
    """Return ((N_kept, n_layers+1, H) float16 array, kept_idx).

    In direct/unspecified mode, every row is kept and
    `kept_idx == list(range(len(df)))`. In cot mode, rows whose cached CoT
    response lacks `####` (or exceeds the model's context window) are skipped.
    Span modes mean-pool over the selected token range so downstream probes can
    keep using one residual vector per prompt/layer.
    """
    df = df.reset_index(drop=True)
    prompts: list[str] = []
    kept_idx: list[int] = []
    # Per-row target position (un-padded token index) or span for modes that
    # read somewhere other than the last non-pad token. None ⇒ use last non-pad.
    target_positions: list[int | tuple[int, int] | None] = []

    if mode == "cot":
        effective_max_length = min(
            max_length,
            int(getattr(model.config, "max_position_embeddings", max_length)),
        )
        skipped_too_long = 0
        for i, row in df.iterrows():
            p = _fmt_cot(row["question"], row.get("original_cot", ""), tokenizer)
            if p is None:
                continue
            prompt_len = len(
                tokenizer(p, truncation=False, add_special_tokens=True)["input_ids"]
            )
            if prompt_len > effective_max_length:
                skipped_too_long += 1
                continue
            prompts.append(p)
            kept_idx.append(int(i))
            target_positions.append(None)
        print(f"  {len(prompts)} / {len(df)} prompts have a usable CoT trace")
        if skipped_too_long:
            print(
                f"  Skipped {skipped_too_long} CoT prompts longer than "
                f"{effective_max_length} tokens"
            )
        if not prompts:
            raise ValueError("No usable CoT prompts remain after filtering.")
        truncate = False
        max_len_kw = None
    elif mode == "direct":
        prompts = [_fmt_direct(q, tokenizer) for q in df["question"].tolist()]
        kept_idx = list(range(len(df)))
        target_positions = [None] * len(prompts)
        truncate = True
        max_len_kw = 1024
    elif mode == "cot_pre_reasoning":
        prompts = [_fmt_cot_pre_reasoning(q, tokenizer) for q in df["question"].tolist()]
        kept_idx = list(range(len(df)))
        target_positions = [None] * len(prompts)
        truncate = True
        max_len_kw = 1024
    elif mode == "question_last_token":
        questions = df["question"].tolist()
        prompts = [_fmt_unspecified(q, tokenizer) for q in questions]
        target_positions = [_last_question_token_index(q, tokenizer) for q in questions]
        kept_idx = list(range(len(df)))
        # Truncation would shift token indices; rely on questions fitting in 1024.
        truncate = False
        max_len_kw = None
    elif mode == "question_span_mean":
        questions = df["question"].tolist()
        prompts = [_fmt_unspecified(q, tokenizer) for q in questions]
        target_positions = [_question_token_span(q, tokenizer) for q in questions]
        kept_idx = list(range(len(df)))
        # Truncation would shift token indices; rely on questions fitting in 1024.
        truncate = False
        max_len_kw = None
    else:
        prompts = [_fmt_unspecified(q, tokenizer) for q in df["question"].tolist()]
        kept_idx = list(range(len(df)))
        target_positions = [None] * len(prompts)
        truncate = True
        max_len_kw = 1024

    # Deduplicate: many datasets (notably gsm_symbolic_pairs) have multiple
    # rows with the same formatted prompt — e.g., positive halves of paired
    # rows from the same underlying instance. Forward-pass once per unique
    # prompt and replicate the result across duplicate rows.
    prompt_to_uniq: dict[str, int] = {}
    inverse_idx: list[int] = []   # for each row index, idx into unique_prompts
    unique_prompts: list[str] = []
    unique_targets: list[int | None] = []
    for p, t in zip(prompts, target_positions):
        idx = prompt_to_uniq.get(p)
        if idx is None:
            idx = len(unique_prompts)
            prompt_to_uniq[p] = idx
            unique_prompts.append(p)
            unique_targets.append(t)
        inverse_idx.append(idx)
    n_dup = len(prompts) - len(unique_prompts)
    if n_dup:
        print(f"  Deduplicated {len(prompts)} prompts -> {len(unique_prompts)} "
              f"unique ({n_dup} duplicates collapsed, "
              f"{100*n_dup/len(prompts):.1f}% forward-pass saving).")

    unique_states: list[np.ndarray] = []
    desc = f"Extracting {mode} residuals"
    for i in tqdm(range(0, len(unique_prompts), batch_size), desc=desc):
        batch = unique_prompts[i : i + batch_size]
        kwargs = dict(return_tensors="pt", padding=True, truncation=truncate)
        if max_len_kw is not None:
            kwargs["max_length"] = max_len_kw
        enc = tokenizer(batch, **kwargs).to(model.device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        for b in range(len(batch)):
            tgt = unique_targets[i + b]
            if tgt is None:
                # Last non-padding token position.
                pos = enc["attention_mask"][b].nonzero()[-1].item()
                states = torch.stack(
                    [layer[b, pos, :].cpu().float() for layer in out.hidden_states],
                    dim=0,
                ).numpy()  # (n_layers+1, H)
            elif isinstance(tgt, tuple):
                # Un-padded [start, stop) span; offset by left-padding count
                # (= first non-pad index, which is 0 for right padding).
                first_non_pad = enc["attention_mask"][b].nonzero()[0].item()
                start, stop = tgt
                start = first_non_pad + start
                stop = first_non_pad + stop
                states = torch.stack(
                    [layer[b, start:stop, :].mean(dim=0).cpu().float()
                     for layer in out.hidden_states],
                    dim=0,
                ).numpy()  # (n_layers+1, H)
            else:
                # Un-padded position; offset by left-padding count (= first
                # non-pad index, which is 0 for right padding).
                first_non_pad = enc["attention_mask"][b].nonzero()[0].item()
                pos = first_non_pad + tgt
                # out.hidden_states: tuple of (n_layers+1) tensors, each (B, T, H)
                states = torch.stack(
                    [layer[b, pos, :].cpu().float() for layer in out.hidden_states],
                    dim=0,
                ).numpy()  # (n_layers+1, H)
            unique_states.append(states)

    # Expand back to one entry per input row by indexing into unique_states.
    all_states = [unique_states[inverse_idx[i]] for i in range(len(prompts))]
    return np.stack(all_states, axis=0).astype(np.float16), kept_idx


def _dataset_suffix(dataset_name: str) -> str:
    return "" if dataset_name == "gsm_symbolic" else f"_{dataset_name}"


def _cache_root(dataset_name: str) -> Path:
    """Root directory for hidden-state caches (.npy and meta CSV)."""
    return CACHE_ROOT_OVERRIDES.get(dataset_name, CACHE_DIR)


def model_mode_dir(model_short: str, mode: str = "direct",
                    dataset_name: str = "gsm_symbolic") -> Path:
    """Output directory for plots and template_similarity result JSONs.
    Lives under the project results tree (OUT_DIR), not on scratch.
    Layout: OUT_DIR/<model_short>/<dataset_name>/<mode>/. The dataset
    subdir was added in the reorg so multi-dataset runs don't collide;
    callers that don't pass a dataset get the gsm_symbolic dir."""
    return OUT_DIR / model_short / dataset_name / mode


def cache_dir(model_short: str, mode: str = "direct",
              dataset_name: str = "gsm_symbolic") -> Path:
    """Directory for the hidden-state cache files."""
    return _cache_root(dataset_name) / model_short / mode


def cache_path(
    model_short: str, correct_only: bool, dataset_name: str = "gsm_symbolic",
    mode: str = "direct",
) -> Path:
    suffix = "_correct" if correct_only else "_all"
    return cache_dir(model_short, mode, dataset_name) / (
        f"hidden_states{_dataset_suffix(dataset_name)}{suffix}.npy"
    )


def meta_path(
    model_short: str, correct_only: bool, dataset_name: str = "gsm_symbolic",
    mode: str = "direct",
) -> Path:
    suffix = "_correct" if correct_only else "_all"
    return cache_dir(model_short, mode, dataset_name) / (
        f"hidden_states{_dataset_suffix(dataset_name)}{suffix}_meta.csv"
    )


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """(N, H) → (N, N) cosine similarity matrix, float32."""
    X = X.astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X_norm = X / norms
    return X_norm @ X_norm.T  # (N, N)


def build_masks(original_ids: np.ndarray):
    """Return boolean within-template and cross-template upper-triangle masks."""
    N = len(original_ids)
    # within[i,j] = True iff same original_id and i < j
    id_col = original_ids[:, None]   # (N, 1)
    id_row = original_ids[None, :]   # (1, N)
    same = (id_col == id_row)        # (N, N)
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    within = same & upper
    cross  = (~same) & upper
    return within, cross


def compute_similarity_stats(
    hidden: np.ndarray,        # (N, n_layers+1, H)
    original_ids: np.ndarray,  # (N,)
    n_cross_sample: int = 0,   # 0 = match within-template pair count
    rng: np.random.Generator = None,
    n_bootstrap: int = 200,
) -> list[dict]:
    """Per-layer within/cross cosine similarity statistics.

    Point estimates and CIs are both at the template level: each template's
    mean similarity is computed first, the across-template mean is the
    reported value, and CIs come from a cluster bootstrap resampling
    templates (with replacement). Pairs within a template are not
    independent — pooling at the pair level both overweights large templates
    and underestimates uncertainty.
    """
    n_layers = hidden.shape[1]
    within_mask, cross_mask = build_masks(original_ids)

    within_idx = np.argwhere(within_mask)  # (P_w, 2)
    cross_idx  = np.argwhere(cross_mask)   # (P_c, 2)

    if rng is None:
        rng = np.random.default_rng(42)
    # Subsample cross pairs to a manageable count (default: match within count)
    if n_cross_sample == 0:
        n_cross_sample = len(within_idx)
    if len(cross_idx) > n_cross_sample:
        sel = rng.choice(len(cross_idx), size=n_cross_sample, replace=False)
        cross_idx = cross_idx[sel]

    # Template index per pair. Within-pairs: t_i == t_j (use either). Cross
    # pairs: anchor on the left endpoint — bootstrap unit is the anchor
    # template's average similarity to the rest.
    uniq_ids, t_inv = np.unique(original_ids, return_inverse=True)
    n_t = len(uniq_ids)
    w_t = t_inv[within_idx[:, 0]]
    c_t = t_inv[cross_idx[:, 0]]
    w_contrib = np.unique(w_t)
    c_contrib = np.unique(c_t)

    print(f"Within-template pairs: {len(within_idx):,} across {len(w_contrib)} templates  |  "
          f"Cross-template pairs (sampled): {len(cross_idx):,} across {len(c_contrib)} anchor templates")

    rows = []
    for L in tqdm(range(n_layers), desc="Computing similarity"):
        sim = cosine_sim_matrix(hidden[:, L, :])

        w_sims = sim[within_idx[:, 0], within_idx[:, 1]].astype(np.float64)
        c_sims = sim[cross_idx[:,  0], cross_idx[:,  1]].astype(np.float64)

        # Per-template means
        w_sum = np.zeros(n_t, dtype=np.float64); w_cnt = np.zeros(n_t, dtype=np.int64)
        np.add.at(w_sum, w_t, w_sims); np.add.at(w_cnt, w_t, 1)
        w_per = np.full(n_t, np.nan, dtype=np.float64)
        w_per[w_cnt > 0] = w_sum[w_cnt > 0] / w_cnt[w_cnt > 0]

        c_sum = np.zeros(n_t, dtype=np.float64); c_cnt = np.zeros(n_t, dtype=np.int64)
        np.add.at(c_sum, c_t, c_sims); np.add.at(c_cnt, c_t, 1)
        c_per = np.full(n_t, np.nan, dtype=np.float64)
        c_per[c_cnt > 0] = c_sum[c_cnt > 0] / c_cnt[c_cnt > 0]

        w_mean = float(np.nanmean(w_per[w_contrib]))
        c_mean = float(np.nanmean(c_per[c_contrib]))

        # Cluster bootstrap over templates
        if len(w_contrib) >= 2 and len(c_contrib) >= 2:
            w_boot = np.empty(n_bootstrap, dtype=np.float64)
            c_boot = np.empty(n_bootstrap, dtype=np.float64)
            for b in range(n_bootstrap):
                w_boot[b] = np.nanmean(w_per[rng.choice(w_contrib, size=len(w_contrib), replace=True)])
                c_boot[b] = np.nanmean(c_per[rng.choice(c_contrib, size=len(c_contrib), replace=True)])
            w_lo, w_hi = np.nanpercentile(w_boot, [2.5, 97.5])
            c_lo, c_hi = np.nanpercentile(c_boot, [2.5, 97.5])
            w_se = float(np.nanstd(w_boot))
            c_se = float(np.nanstd(c_boot))
        else:
            w_lo = w_hi = c_lo = c_hi = float("nan")
            w_se = c_se = float("nan")

        rows.append({
            "layer":         L,
            "within_mean":   w_mean,
            "within_std":    float(np.nanstd(w_per[w_contrib])),
            "within_se":     w_se,
            "within_ci_lo":  float(w_lo),
            "within_ci_hi":  float(w_hi),
            "cross_mean":    c_mean,
            "cross_std":     float(np.nanstd(c_per[c_contrib])),
            "cross_se":      c_se,
            "cross_ci_lo":   float(c_lo),
            "cross_ci_hi":   float(c_hi),
            "gap":           float(w_mean - c_mean),
            "gap_se":        float(np.sqrt(w_se**2 + c_se**2)) if np.isfinite(w_se) else float("nan"),
            "n_within":      int(len(w_sims)),
            "n_cross":       int(len(c_sims)),
            "n_within_templates": int(len(w_contrib)),
            "n_cross_templates":  int(len(c_contrib)),
        })

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def find_large_drop_bands(
    df: pd.DataFrame,
    z_threshold: float = 3.0,
    min_drop: float = 0.025,
    nms_window: int = 3,
) -> list[dict]:
    """Find material, statistically significant adjacent drops by curve.

    Defaults are tuned so that the weaker abstraction signal in CoT mode
    (L21-L22 cross drop ~0.03) gets annotated, while a non-maximum-suppression
    pass with window `nms_window` prevents nearby layers from all producing
    bands (e.g. direct's L21-L24 would otherwise yield 4 stacked bands; NMS
    keeps just the layer with the biggest drop in each window).
    """
    if len(df) < 2:
        return []

    layers = df["layer"].to_numpy()
    cross = df["cross_mean"].to_numpy(dtype=float)
    cross_se = df["cross_se"].to_numpy(dtype=float)
    within = df["within_mean"].to_numpy(dtype=float)
    within_se = df["within_se"].to_numpy(dtype=float)

    cross_drop = cross[:-1] - cross[1:]
    cross_drop_se = np.sqrt(cross_se[:-1] ** 2 + cross_se[1:] ** 2)
    cross_z = np.divide(
        cross_drop,
        cross_drop_se,
        out=np.zeros_like(cross_drop, dtype=float),
        where=cross_drop_se > 0,
    )

    within_drop = within[:-1] - within[1:]
    within_drop_se = np.sqrt(within_se[:-1] ** 2 + within_se[1:] ** 2)
    within_z = np.divide(
        within_drop,
        within_drop_se,
        out=np.zeros_like(within_drop, dtype=float),
        where=within_drop_se > 0,
    )

    candidates = []
    for i in range(len(layers) - 1):
        cross_sig = cross_drop[i] >= min_drop and cross_z[i] >= z_threshold
        within_sig = within_drop[i] >= min_drop and within_z[i] >= z_threshold
        if not cross_sig and not within_sig:
            continue
        if cross_sig and within_sig:
            band_type = "both"
        elif cross_sig:
            band_type = "cross_only"
        else:
            band_type = "within_only"
        # Score: largest drop (across the two curves that are significant) —
        # used for non-maximum suppression below.
        score = max(
            cross_drop[i] if cross_sig else 0.0,
            within_drop[i] if within_sig else 0.0,
        )
        candidates.append({
            "i": i,
            "transition": (int(layers[i]), int(layers[i + 1])),
            "center": float((layers[i] + layers[i + 1]) / 2),
            "start": float(layers[i]),
            "end": float(layers[i + 1]),
            "type": band_type,
            "cross_z": float(cross_z[i]),
            "within_z": float(within_z[i]),
            "cross_drop": float(cross_drop[i]),
            "within_drop": float(within_drop[i]),
            "score": float(score),
        })

    # Non-maximum suppression: keep candidates in descending score order,
    # rejecting any that lie within `nms_window` of an already-kept band.
    candidates.sort(key=lambda c: -c["score"])
    kept = []
    for c in candidates:
        if all(abs(c["i"] - k["i"]) >= nms_window for k in kept):
            kept.append(c)

    # Re-sort by layer for plot ordering.
    kept.sort(key=lambda c: c["i"])
    for c in kept:
        c.pop("i", None)
        c.pop("score", None)
    return kept


def plot_results(
    rows: list[dict],
    identifier: str,
    correct_only: bool,
    out_dir: Path,
    mode: str = "direct",
    model_short: str = "",
):
    df = pd.DataFrame(rows)
    layers = df["layer"].values
    suffix = "correct only" if correct_only else "all instances"
    suffix = f"{mode}, {suffix}"
    trend_bands = find_large_drop_bands(df)
    fsuffix = "_correct" if correct_only else "_all"
    # Stage boundaries are derived per-model from the largest drops
    # detected by find_large_drop_bands, instead of being hard-coded for
    # the 80-layer Llama-3.3-70B network. We pick the top-2 cross_only
    # drops by magnitude (Template-structure emergence + consolidation),
    # the top-1 within-or-both drop (Instance-specific binding), and the
    # final layer as Output readout.
    last_layer = int(df["layer"].iloc[-1])
    cross_only_bands = [b for b in trend_bands if b["type"] == "cross_only"]
    binding_bands   = [b for b in trend_bands if b["type"] in ("both", "within_only")]
    cross_only_bands.sort(key=lambda b: -b["cross_drop"])
    top_cross = sorted(cross_only_bands[:2], key=lambda b: b["end"])
    binding_bands.sort(
        key=lambda b: -max(b.get("within_drop", 0.0), b.get("cross_drop", 0.0))
    )
    top_binding = binding_bands[:1]
    stage_boundaries: list[tuple[int, str, str]] = []
    cross_labels = ["Template-structure\nemergence", "Template-structure\nconsolidation"]
    for band, label in zip(top_cross, cross_labels):
        stage_boundaries.append((int(band["end"]), label, "cross"))
    for band in top_binding:
        # Skip duplicates with already-marked cross bands.
        if any(b[0] == int(band["end"]) for b in stage_boundaries):
            continue
        stage_boundaries.append((int(band["end"]), "Instance-specific\nbinding", "both"))
    stage_boundaries.append((last_layer, "Output\nreadout", "both"))
    # De-dup by layer (in case binding lands on the last layer); keep first
    # label assigned.
    seen = set(); deduped = []
    for layer, label, kind in stage_boundaries:
        if layer in seen:
            continue
        seen.add(layer); deduped.append((layer, label, kind))
    stage_boundaries = sorted(deduped, key=lambda b: b[0])
    boundary_style = {
        "cross": {"color": "#d6604d", "ls": "--"},
        "both":  {"color": "#333333", "ls": "-."},
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 4.8))

    # Prefer bootstrap percentile CIs when present (template-level cluster
    # bootstrap); fall back to ±2σ for legacy result files without ci_lo/ci_hi.
    if "within_ci_lo" in df.columns:
        w_lo = df["within_ci_lo"]; w_hi = df["within_ci_hi"]
        c_lo = df["cross_ci_lo"];  c_hi = df["cross_ci_hi"]
    else:
        w_lo = df["within_mean"] - 2 * df["within_se"]
        w_hi = df["within_mean"] + 2 * df["within_se"]
        c_lo = df["cross_mean"]  - 2 * df["cross_se"]
        c_hi = df["cross_mean"]  + 2 * df["cross_se"]
    ax.fill_between(layers, w_lo, w_hi, alpha=0.2, color="#2166ac")
    ax.fill_between(layers, c_lo, c_hi, alpha=0.2, color="#d6604d")
    ax.plot(layers, df["within_mean"], color="#2166ac", lw=1.8,
            label="Within template")
    ax.plot(layers, df["cross_mean"],  color="#d6604d", lw=1.8,
            label="Cross template")
    ax.set_ylabel("Mean pairwise cosine similarity")

    stage_ticks: list[int] = []
    layer_to_idx = {int(l): i for i, l in enumerate(layers)}
    for boundary, label, kind in stage_boundaries:
        style = boundary_style[kind]
        ax.axvline(boundary, color=style["color"], lw=0.9, alpha=0.7, ls=style["ls"])
        ax.text(
            boundary, 0.055, label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", rotation=90, fontsize=7.5, color=style["color"],
        )
        stage_ticks.append(boundary)
        # Annotate cosine value at each boundary on both curves so drops and
        # plateaus are easy to read off the figure.
        idx = layer_to_idx.get(int(boundary))
        if idx is not None:
            wv = float(df["within_mean"].iloc[idx])
            cv = float(df["cross_mean"].iloc[idx])
            ax.scatter([boundary, boundary], [wv, cv],
                       s=18, color=["#2166ac", "#d6604d"], zorder=5,
                       edgecolor="white", linewidth=0.6)
            bbox = dict(boxstyle="round,pad=0.15", fc="white",
                        ec="none", alpha=0.85)
            ax.annotate(f"{wv:.2f}", (boundary, wv),
                        xytext=(5, 7), textcoords="offset points",
                        fontsize=7.5, color="#2166ac", fontweight="bold",
                        bbox=bbox, zorder=6)
            ax.annotate(f"{cv:.2f}", (boundary, cv),
                        xytext=(5, -11), textcoords="offset points",
                        fontsize=7.5, color="#d6604d", fontweight="bold",
                        bbox=bbox, zorder=6)
    if not trend_bands:
        ax.text(
            0.5, 0.94,
            "No significant adjacent drops (drop >= 0.025, z >= 3, NMS w=3)",
            transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="#555555",
        )
    from matplotlib.lines import Line2D
    sim_handles, sim_labels = ax.get_legend_handles_labels()
    boundary_handles = [
        Line2D([0], [0], color=boundary_style["cross"]["color"],
               ls=boundary_style["cross"]["ls"], lw=1.2,
               label="Cross-only drop"),
        Line2D([0], [0], color=boundary_style["both"]["color"],
               ls=boundary_style["both"]["ls"], lw=1.2,
               label="Within + cross drop"),
    ]
    ax.legend(
        handles=sim_handles + boundary_handles,
        fontsize=9.5,
        loc="lower left",
        frameon=True,
    )
    ax.set_xlabel("Layer")
    # Custom x-axis ticks: regular interval marks plus the per-model
    # stage-boundary layers. Stage layers are bolded so the reader can
    # read them off the x-axis without needing inline annotations (stage
    # names live in the figure caption instead). Drops default-interval
    # ticks that sit within 4 layers of a stage tick to avoid overlap.
    n_layers_plot = int(layers[-1]) + 1
    # Pick a tick interval that gives ~4-6 default ticks regardless of
    # model depth (28-layer Qwen-Math-7B through 80-layer Llama-70B).
    interval = max(5, n_layers_plot // 6)
    default_ticks = list(range(0, n_layers_plot, interval))
    if (n_layers_plot - 1) not in default_ticks:
        default_ticks.append(n_layers_plot - 1)
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

    plt.tight_layout()
    plot_path = out_dir / f"template_similarity{identifier}{fsuffix}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8), sharex=True)

    # ── Middle panel: per-curve layer-wise change (within and cross) ─────────
    # Shows the actual mechanism behind the gap derivative: at each layer,
    # how much within-template similarity dropped vs how much cross-template
    # similarity dropped. Bars point down because the curves mostly drop.
    ax2 = axes[0]
    within_arr = df["within_mean"].values
    cross_arr = df["cross_mean"].values
    d_within = np.diff(within_arr, prepend=within_arr[0])
    d_cross  = np.diff(cross_arr,  prepend=cross_arr[0])
    bar_w = 0.42
    ax2.bar(layers - bar_w/2, d_within, width=bar_w, color="#2166ac",
            label="Within template")
    ax2.bar(layers + bar_w/2, d_cross,  width=bar_w, color="#d6604d",
            label="Cross template")
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylabel("Layer-wise change\n(this layer − previous layer)")
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax2.legend(title="Pair type", fontsize=9, title_fontsize=9)
    # The gap derivative is still needed below, so compute it here from the
    # already-loaded gap series for the bottom panel.
    gap_arr = df["gap"].values

    # ── Bottom panel: per-layer change in the gap ────────────────────────────
    # Each bar shows gap[L] - gap[L-1]. Positive: cross-template dropped more
    # than within-template at this layer. Negative: within-template dropped
    # more than cross-template. We deliberately do NOT map the sign onto
    # specific stages — both binding and computation involve both curves
    # dropping, and which drops faster isn't a clean stage diagnostic.
    ax3 = axes[1]
    d_gap = np.diff(gap_arr, prepend=gap_arr[0])
    pos_color = "#1b7837"  # green
    neg_color = "#9970ab"  # purple
    d_colors = [pos_color if v > 0 else neg_color for v in d_gap]
    ax3.bar(layers, d_gap, color=d_colors, width=1.0, edgecolor="none")
    ax3.axhline(0, color="grey", linewidth=0.8)
    ax3.set_ylabel("Change in gap\n(this layer − previous layer)")
    ax3.set_xlabel("Layer")
    ax3.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax3.legend(
        handles=[
            mpatches.Patch(
                color=pos_color, alpha=0.9,
                label="Different-template similarity drops more",
            ),
            mpatches.Patch(
                color=neg_color, alpha=0.9,
                label="Same-template similarity drops more",
            ),
        ],
        fontsize=8.5, loc="best",
    )
    # Annotate the top few positive layers to highlight abstraction-style
    # transitions that aren't picked up by the absolute-drop band threshold.
    pos_idx = np.argsort(d_gap)[-5:][::-1]
    for i in pos_idx:
        if d_gap[i] <= 0:
            continue
        ax3.annotate(
            f"L{int(layers[i])}",
            xy=(layers[i], d_gap[i]),
            xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=8, color=pos_color,
        )

    plt.tight_layout()
    diagnostic_path = out_dir / f"template_similarity{identifier}{fsuffix}_diagnostics.png"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(diagnostic_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Diagnostic plot saved to {diagnostic_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    model_id: str,
    dataset_name: str,
    correct_only: bool,
    batch_size: int,
    n_cross_sample: int,
    seed: int,
    mode: str = "direct",
    cache_only: bool = False,
):
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    mode_dir = model_mode_dir(model_short, mode, dataset_name)
    if not cache_only:
        mode_dir.mkdir(parents=True, exist_ok=True)
    cache_dir(model_short, mode, dataset_name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    correctness_col = _correctness_col(mode)

    # ── Load all-instances data (correctness filter applied later, in view) ──
    df_all = load_gsm_dataset(
        model_id, dataset_name=dataset_name, correct_only=False, mode=mode,
    )
    print(
        f"Loaded {len(df_all)} {DATASET_NAMES[dataset_name]} instances "
        f"from {df_all['original_id'].nunique()} templates (mode={mode})"
    )

    # ── All-instances hidden states are the source of truth ──────────────────
    all_hs_path = cache_path(model_short, correct_only=False,
                              dataset_name=dataset_name, mode=mode)
    all_mt_path = meta_path(model_short, correct_only=False,
                              dataset_name=dataset_name, mode=mode)

    if all_hs_path.exists() and all_mt_path.exists():
        print(f"Loading cached all-instances hidden states from {all_hs_path}")
        hidden_all = np.load(all_hs_path)
        cached_meta = pd.read_csv(all_mt_path)
        if hidden_all.shape[0] != len(cached_meta):
            raise ValueError(
                f"Cache shape mismatch: {hidden_all.shape[0]} rows in "
                f"{all_hs_path} vs. {len(cached_meta)} in meta."
            )
        # The cache may be a strict subset of the current dataset — e.g. in
        # cot mode, extraction drops rows that lack a usable CoT trace. Accept
        # any cached subset of current_keys; error only if the cache contains
        # keys that no longer exist in the current dataset (stale cache).
        #
        # Paired datasets have many rows per (original_id, instance), so the
        # cache alignment key must include pair identity and the row value.
        # Otherwise a cache reload can expand/misorder rows and silently break
        # downstream paired-CV probes.
        if dataset_name == "gsm_symbolic_pairs":
            align_cols = [
                c for c in PAIRED_ALIGN_KEYS
                if c in cached_meta.columns and c in df_all.columns
            ]
            missing_cols = [c for c in PAIRED_ALIGN_KEYS if c not in align_cols]
            if missing_cols:
                raise ValueError(
                    f"Paired cache meta is missing alignment columns {missing_cols}; "
                    "delete/rebuild the cache so pair_id, label, and value are "
                    "recorded."
                )
        else:
            align_cols = ["original_id", "instance"]

        if cached_meta.duplicated(align_cols).any():
            raise ValueError(
                f"Cache meta has duplicate alignment keys for {align_cols}; "
                "delete/rebuild the cache."
            )
        if df_all.duplicated(align_cols).any():
            raise ValueError(
                f"Current dataset has duplicate alignment keys for {align_cols}; "
                "the cache cannot be aligned safely."
            )

        cached_keys = list(cached_meta[align_cols].itertuples(index=False, name=None))
        current_keys = list(df_all[align_cols].itertuples(index=False, name=None))
        current_keys_set = set(current_keys)
        missing_in_current = [k for k in cached_keys if k not in current_keys_set]
        if missing_in_current:
            raise ValueError(
                f"Cache at {all_hs_path} contains {len(missing_in_current)} "
                "rows not present in the current dataset (e.g. "
                f"{missing_in_current[:3]}). Delete the cache and rerun."
            )
        if len(cached_keys) != len(current_keys):
            print(
                f"  Cache is a subset of the current dataset: "
                f"{len(cached_keys)}/{len(current_keys)} rows. Aligning df to cache."
            )
        # Align df_all to the cache's row order so the in-memory df and the
        # cached residuals stay row-indexed together.
        df_all = (df_all
                  .set_index(align_cols)
                  .loc[cached_keys]
                  .reset_index())
    else:
        print(f"Extracting hidden states for {len(df_all)} instances (mode={mode})...")
        if torch is None or tqdm is None:
            raise ImportError(
                "Full extraction requires torch and tqdm. Install project "
                "dependencies, or use --plot_only with existing JSON results."
            )
        from utils.noop_utils import load_model

        model, tokenizer = load_model(model_id)
        hidden_all, kept_idx = extract_hidden_states(
            df_all, model, tokenizer, batch_size=batch_size, mode=mode,
        )
        if mode == "cot" and len(kept_idx) != len(df_all):
            df_all = df_all.iloc[kept_idx].reset_index(drop=True)
        print(f"Extracted: {hidden_all.shape}")
        np.save(all_hs_path, hidden_all)
        # Carry through any join columns that downstream probes might need.
        # Always include the mode's correctness column; also include the OTHER
        # correctness column (so a single cache supports cross-mode filtering)
        # and any pair-structure columns (pair_id, label, role_type, role,
        # target_value, value) when present, so paired-CV downstream doesn't
        # need to re-join with the inference CSVs.
        keep_cols = ["original_id", "instance", correctness_col]
        for opt in [
            "original_cot_correctness", "original_direct_correctness",
            "pair_id", "label", "role_type", "role",
            "target_value", "value",
        ]:
            if opt in df_all.columns and opt not in keep_cols:
                keep_cols.append(opt)
        meta_df = df_all[keep_cols].copy()
        meta_df.insert(0, "dataset", dataset_name)
        meta_df.to_csv(all_mt_path, index=False)
        print(f"Cached to {all_hs_path}  (meta cols: {list(meta_df.columns)})")
        del model

    # ── Cache-only short-circuit ─────────────────────────────────────────────
    # Skip the within/cross similarity stats and plotting if the caller only
    # wants the hidden-state cache (e.g., gsm_symbolic_pairs is consumed by
    # downstream probes only and the similarity analysis isn't meaningful on
    # paired data — see the note in the entity-pairs design discussion).
    if cache_only:
        print("cache_only set; skipping similarity computation and plotting.")
        return

    # ── Derive correct-only view by filtering (in-memory only) ───────────────
    # Downstream tools load the all-cache and apply the same filter themselves,
    # so we don't write a separate hidden_states_correct.npy here.
    if correct_only:
        correct_mask = _coerce_bool(df_all[correctness_col]).fillna(False).to_numpy(dtype=bool)
        hidden = hidden_all[correct_mask]
        df = df_all[correct_mask].reset_index(drop=True)
        print(
            f"Filtered to {len(df)}/{len(df_all)} correct instances "
            f"(column={correctness_col})"
        )
    else:
        hidden = hidden_all
        df = df_all

    original_ids = df["original_id"].values

    # ── Compute similarities ──────────────────────────────────────────────────
    rows = compute_similarity_stats(
        hidden, original_ids, n_cross_sample=n_cross_sample, rng=rng
    )

    # ── Save results ──────────────────────────────────────────────────────────
    # Filenames no longer encode the dataset — the mode_dir already contains
    # a per-dataset subdir (see model_mode_dir).
    fsuffix = "_correct" if correct_only else "_all"
    results_path = mode_dir / f"template_similarity{fsuffix}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Results saved to {results_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    rdf = pd.DataFrame(rows)
    peak_layer = rdf.loc[rdf["gap"].idxmax(), "layer"]
    peak_gap   = rdf["gap"].max()
    close_layer = rdf.loc[rdf["gap"] < 0.01, "layer"].min() if (rdf["gap"] < 0.01).any() else "never"
    print(f"\nPeak gap {peak_gap:.4f} at layer {peak_layer}")
    print(f"Gap drops below 0.01 at layer {close_layer}  ← Stage 2→3 boundary estimate")
    print(f"\nPer-bin gap summary:")
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]:
        m = rdf[(rdf["layer"] >= lo) & (rdf["layer"] < hi)]["gap"].mean()
        print(f"  L{lo:>2}–{hi:>2}: mean gap = {m:+.4f}")

    plot_results(
        rows, "", correct_only,
        mode_dir, mode=mode, model_short=model_short,
    )


def plot_only(model_id: str, dataset_name: str, correct_only: bool, mode: str = "direct"):
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    mode_dir = model_mode_dir(model_short, mode, dataset_name)
    fsuffix = "_correct" if correct_only else "_all"
    results_path = mode_dir / f"template_similarity{fsuffix}.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No results at {results_path}. Run without --plot_only first.")
    with open(results_path) as f:
        rows = json.load(f)
    plot_results(
        rows, "", correct_only,
        mode_dir, mode=mode, model_short=model_short,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--dataset",
        default="gsm_symbolic",
        choices=sorted(DATASET_NAMES),
        help="Which GSM variant to extract/cache.",
    )
    parser.add_argument(
        "--correct_only", action="store_true",
        help="Use only instances correct under the selected mode's available "
             "correctness signal (CoT uses original_cot_correctness; "
             "direct/unspecified use original_direct_correctness).",
    )
    parser.add_argument(
        "--n_cross_sample", type=int, default=0,
        help="Cross-template pairs to sample (0 = match within-template count)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["direct", "cot", "cot_pre_reasoning", "unspecified",
                 "question_last_token", "question_span_mean"],
        default="direct",
        help="direct: residual at the answer-prefix token of the direct prompt. "
             "cot: residual at the answer-prefix token following the cached "
             "chain-of-thought reasoning (requires `original_cot` in eval CSVs). "
             "cot_pre_reasoning: cot-framed prompt with the CoT instruction in "
             "place but no cached reasoning; residual is read at the final "
             "token (assistant-header newline), i.e. just before the model "
             "would generate the first reasoning token. "
             "unspecified: residual at the last prompt token when the user asks "
             "only the question, with no direct or CoT instruction. "
             "question_last_token: residual at the last token of the question "
             "itself inside the chat-formatted prompt (before the user-turn "
             "closing tokens and assistant header). "
             "question_span_mean: mean-pooled residual over all question tokens "
             "inside that same chat-formatted prompt.",
    )
    parser.add_argument(
        "--cache_only", action="store_true",
        help="Extract and save the hidden-state cache, then exit. Skip the "
             "within/cross template similarity computation and plotting. "
             "Useful when the dataset is consumed only by downstream probes "
             "(e.g., gsm_symbolic_pairs) and the similarity analysis isn't "
             "meaningful or wanted. Auto-on for gsm_symbolic_pairs.",
    )
    args = parser.parse_args()

    # Auto-cache-only for the pairs dataset: similarity stats on paired data
    # are confounded by within-pair contamination (see entity-pairs design
    # discussion); the cache is the only useful artifact downstream.
    cache_only = args.cache_only or args.dataset == "gsm_symbolic_pairs"

    if args.plot_only:
        plot_only(args.model_id, args.dataset, args.correct_only, mode=args.mode)
    else:
        run(
            model_id=args.model_id,
            dataset_name=args.dataset,
            correct_only=args.correct_only,
            batch_size=args.batch_size,
            n_cross_sample=args.n_cross_sample,
            seed=args.seed,
            mode=args.mode,
            cache_only=cache_only,
        )


if __name__ == "__main__":
    main()
