"""
Binding-crystallization experiment at the answer position.

For each gsm_symbolic question, identify the numerals that are bound to
symbolic variables (the *operands* parsed from `symbol_binding`), and at
every layer project the answer-position residual through the unembedding to
read off the rank / log-probability the model's current state assigns to
those operand token IDs.

Direct test of the four-stage hypothesis prediction that Variable Binding
crystallizes around L36–L40 in Llama-3.3-70B: the layer at which the answer
position becomes preferentially sensitive to *this question's* bound operands
is the empirical onset of binding.

Two baselines:
  - number_only:      non-operand numerals from the same question.  Isolates
                      bound-numeral preference from generic numeral affinity.
  - operand_shuffled: operand groups taken from a different (permuted) prompt.
                      A cross-prompt null — positive contrast means the
                      residual is question-specific, not generic.

Supports two inference modes:
  - direct (default): residual at the position right after the prompt's
    `#### ` answer prefix.  Reuses cached hidden states from
    run_template_similarity.py — no extra forward passes needed.
  - cot: residual at the position right after `#### ` *following* the model's
    own cached chain-of-thought reasoning (column `original_cot` in the eval
    CSVs).  Requires a fresh forward pass; results are cached in OUT_DIR for
    subsequent runs.

Usage:
    python run_input_recovery.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_input_recovery.py --mode cot --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_input_recovery.py --mode cot --plot_only
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import (
    ANSWER_PREFIX, CACHE_DIR, COT_INSTRUCTION, DIRECT_INSTRUCTION,
    LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP,
)
from utils.noop_utils import load_model, parse_symbol_bindings

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "input_recovery"
HS_DIR  = Path(LOGIT_LENS_RESULT_DIR).parent / "template_similarity"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths import RESULT_DIR


CATEGORIES = ["operand", "number_only", "operand_shuffled"]

# Llama-3.3-70B (80 layers) — binding band triangulated from op-multiset
# probe decline, patching cliff, and template-similarity peak.
BINDING_BAND = (36, 40)

# Numeric literals in the question text.
NUMBER_RE = re.compile(
    r"(?<!\w)[+-]?\d[\d,]*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?(?!\w)"
)


# ---------------------------------------------------------------------------
# Token-group extraction
# ---------------------------------------------------------------------------

def operand_groups_from_binding(question: str, symbol_binding: str, tokenizer):
    """Return token-ID groups and char spans for operand values in symbol_binding.

    Tracks values bound to symbolic variables, e.g. `x = 210, y = 7, z = 72`.

    Returns:
        groups: list[list[int]] — vocab token IDs per bound operand value.
        all_spans: list[tuple[int, int]] — char spans of every operand
            occurrence; used to subtract operands from `number_only`.
    """
    bindings = parse_symbol_bindings(symbol_binding)
    if not bindings:
        return [], []

    enc = tokenizer(question, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    groups = []
    all_spans = []
    seen_values = set()

    for raw_value in bindings.values():
        value = str(raw_value).strip()
        if not value or value in seen_values:
            continue
        seen_values.add(value)

        pattern = re.compile(rf"(?<![\w.]){re.escape(value)}(?![\w.])")
        spans = [(m.start(), m.end()) for m in pattern.finditer(question)]
        if not spans and "," in value:
            no_comma = value.replace(",", "")
            pattern = re.compile(rf"(?<![\w.]){re.escape(no_comma)}(?![\w.])")
            spans = [(m.start(), m.end()) for m in pattern.finditer(question)]
        if not spans:
            continue

        tok_group = []
        for tok_id, (s, e) in zip(token_ids, offsets):
            if s == e:
                continue
            if any(max(s, ss) < min(e, ee) for ss, ee in spans):
                tok_group.append(tok_id)
        if tok_group:
            groups.append(tok_group)
            all_spans.extend(spans)

    return groups, all_spans


def number_only_groups(question: str, tokenizer, operand_spans) -> list:
    """Token-ID groups for numerals in the question that are NOT bound operands.

    Same-question control: isolates bound-numeral preference from generic
    numeral affinity.  Empty when every numeral is also an operand.
    """
    enc = tokenizer(question, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    def overlaps_operand(s, e):
        return any(max(s, ss) < min(e, ee) for ss, ee in operand_spans)

    groups = []
    seen_keys = set()
    for m in NUMBER_RE.finditer(question):
        s, e = m.start(), m.end()
        if overlaps_operand(s, e):
            continue
        key = m.group(0).lower().strip()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        tok_group = [
            tok_id for tok_id, (ts, te) in zip(token_ids, offsets)
            if ts != te and max(s, ts) < min(e, te)
        ]
        if tok_group:
            groups.append(tok_group)
    return groups


# ---------------------------------------------------------------------------
# Prompt construction and CoT hidden-state extraction
# ---------------------------------------------------------------------------

ANSWER_MARKER = "####"
COT_CACHE_VERSION = 2
COT_MAX_LENGTH = 4096


def _cot_cache_fingerprint(tokenizer) -> str:
    payload = json.dumps(
        {
            "answer_prefix": ANSWER_PREFIX,
            "cache_version": COT_CACHE_VERSION,
            "cot_instruction": COT_INSTRUCTION,
            "tokenizer": getattr(tokenizer, "name_or_path", ""),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def format_direct_prompt(question: str, tokenizer) -> str:
    """Direct prompt: chat-formatted question + DIRECT_INSTRUCTION, ending at
    ANSWER_PREFIX so the last token is right before the answer integer."""
    msgs = [{"role": "user", "content": f"{question}\n{DIRECT_INSTRUCTION}"}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    ) + ANSWER_PREFIX


def format_cot_prompt(question: str, cot_response: str, tokenizer):
    """CoT prompt: chat-formatted question + COT_INSTRUCTION, followed by the
    model's cached chain-of-thought reasoning up to and including `#### ` (the
    answer prefix).  The last token is right before the answer integer — the
    analog of the direct-mode answer position.

    Returns None if `cot_response` does not contain the answer marker.
    """
    if not isinstance(cot_response, str) or ANSWER_MARKER not in cot_response:
        return None

    # Take everything up to the LAST occurrence of `####` and end with the
    # canonical "#### " prefix so the answer position is consistent across
    # prompts regardless of trailing whitespace in the cached response.
    reasoning_prefix = cot_response.rsplit(ANSWER_MARKER, 1)[0].rstrip()

    msgs = [{"role": "user", "content": f"{question}\n{COT_INSTRUCTION}"}]
    assistant_open = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
    )
    return f"{assistant_open}{reasoning_prefix}\n\n{ANSWER_PREFIX}"


def extract_cot_hidden_states(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int = 2,
    max_length: int = COT_MAX_LENGTH,
) -> tuple:
    """Forward each prompt through the model and gather the last-token residual
    at every layer.  Operates on rows for which `format_cot_prompt` succeeded.

    Returns:
        states: (N, n_layers+1, H) float16 array — residual at the answer
            position (the token right after `#### `).
        kept_idx: list[int] — df row indices that produced a valid prompt
            (i.e. had `####` in the cached CoT response).
    """
    prompts = []
    kept_idx = []
    skipped_too_long = 0
    model_max_length = getattr(model.config, "max_position_embeddings", None)
    effective_max_length = (
        min(max_length, int(model_max_length))
        if model_max_length is not None else max_length
    )
    for i, row in df.reset_index(drop=True).iterrows():
        p = format_cot_prompt(row["question"], row.get("original_cot", ""), tokenizer)
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

    print(f"  {len(prompts)} / {len(df)} prompts have a usable CoT trace")
    if skipped_too_long:
        print(
            f"  Skipped {skipped_too_long} CoT prompts longer than "
            f"{effective_max_length} tokens"
        )
    if not prompts:
        raise ValueError("No usable CoT prompts remain after filtering.")

    all_states = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting CoT residuals"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        for b in range(len(batch)):
            last = enc["attention_mask"][b].nonzero()[-1].item()
            states = torch.stack(
                [layer[b, last, :].cpu().float() for layer in out.hidden_states],
                dim=0,
            ).numpy()
            all_states.append(states)

    return np.stack(all_states, axis=0).astype(np.float16), kept_idx


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def decode_per_layer(
    hidden_states: np.ndarray,
    model,
    prompt_cats: list,
    batch_size: int = 64,
) -> tuple:
    """Project the answer-position residual at every layer through final norm
    + lm_head, then per prompt and per category record:

      - mean log-probability the distribution assigns to the category's
        token groups, and
      - mean rank (0 = top of vocab) of those groups.

    Returns:
        logp: (n_layers, N, n_categories) mean log-prob, NaN where empty.
        rank: (n_layers, N, n_categories) mean rank (0 = top).
    """
    N, n_layers, H = hidden_states.shape
    n_cats = len(CATEGORIES)
    logp_out = np.full((n_layers, N, n_cats), np.nan, dtype=np.float32)
    rank_out = np.full((n_layers, N, n_cats), np.nan, dtype=np.float32)

    norm    = model.model.norm
    lm_head = model.lm_head

    def module_device(module):
        try:
            return next(module.parameters()).device
        except StopIteration:
            return next(module.buffers()).device

    norm_device = module_device(norm)
    head_device = module_device(lm_head)
    dtype       = norm.weight.dtype

    cat_token_groups = [
        [pc.get(cat, []) for cat in CATEGORIES] for pc in prompt_cats
    ]

    for L in tqdm(range(n_layers), desc="Decoding per layer"):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(hidden_states[start:end, L, :]).to(
                device=norm_device, dtype=dtype
            )

            with torch.no_grad():
                normed = norm(batch)
                if normed.device != head_device:
                    normed = normed.to(head_device)
                logits    = lm_head(normed).float()
                log_probs = torch.log_softmax(logits, dim=-1)

                sorted_idx = logits.argsort(dim=-1, descending=True)
                arange_v   = torch.arange(
                    logits.shape[1], device=logits.device, dtype=torch.long,
                ).unsqueeze(0).expand_as(sorted_idx)
                ranks = torch.empty_like(sorted_idx)
                ranks.scatter_(1, sorted_idx, arange_v)

            for i, prompt_idx in enumerate(range(start, end)):
                for ci, token_groups in enumerate(cat_token_groups[prompt_idx]):
                    if not token_groups:
                        continue
                    logp_group_scores = []
                    rank_group_scores = []
                    for tok_ids in token_groups:
                        if not tok_ids:
                            continue
                        idx = torch.as_tensor(tok_ids, device=log_probs.device)
                        logp_group_scores.append(log_probs[i, idx].mean().item())
                        rank_group_scores.append(ranks[i, idx].float().mean().item())
                    if logp_group_scores:
                        logp_out[L, prompt_idx, ci] = float(np.mean(logp_group_scores))
                        rank_out[L, prompt_idx, ci] = float(np.mean(rank_group_scores))

    return logp_out, rank_out


# ---------------------------------------------------------------------------
# Data alignment
# ---------------------------------------------------------------------------

def load_gsm_symbolic_df(model_id: str) -> pd.DataFrame:
    """Concatenate all gsm_symbolic result CSVs for this model.

    Tries the legacy vLLM path first, then falls back to the
    transformers_direct path used by the per-set inference runner for
    new models. Either layout is acceptable; only the column schema is
    relied on downstream.
    """
    file_suffix = model_id.split("/")[-1]
    vllm_pattern = os.path.join(
        RESULT_DIR, "gsm_symbolic",
        f"gsm_symbolic_set_*_results_{file_suffix}.csv",
    )
    tfm_pattern = os.path.join(
        RESULT_DIR, "transformers_direct", "gsm_symbolic",
        f"gsm_symbolic_set_*_results_{file_suffix}_transformers.csv",
    )
    files = sorted(glob.glob(vllm_pattern))
    if not files:
        files = sorted(glob.glob(tfm_pattern))
    if not files:
        raise FileNotFoundError(
            f"No GSM-Symbolic files for {file_suffix}:\n"
            f"  tried vllm:  {vllm_pattern}\n"
            f"  tried tfm:   {tfm_pattern}"
        )
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_aligned_questions(model_id: str, meta_path: Path) -> pd.DataFrame:
    """Re-load gsm_symbolic and reorder rows to match the cached hidden-states
    ordering recorded in the metadata CSV (direct mode)."""
    df = load_gsm_symbolic_df(model_id)
    meta = pd.read_csv(meta_path)
    df_aligned = (
        meta[["original_id", "instance"]]
        .merge(df, on=["original_id", "instance"], how="left")
        .reset_index(drop=True)
    )
    assert len(df_aligned) == len(meta), "Alignment mismatch"
    assert df_aligned["question"].notna().all(), "Some questions missing after merge"
    return df_aligned


# ---------------------------------------------------------------------------
# Mode-aware paths
# ---------------------------------------------------------------------------

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


def out_paths(model_short: str, mode: str, correct_only: bool = False) -> dict:
    """Bundle of output file paths for a given (model, mode, correctness).

    Layout:  OUT_DIR / <model_short> / <files>
    """
    base = OUT_DIR / model_short
    base.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_{'correct' if correct_only else 'all'}"
    return {
        "npz":         base / f"input_recovery_{tag}.npz",
        "meta":        base / f"input_recovery_{tag}_meta.json",
        "summary":     base / f"operand_movement_{tag}.json",
        "plot_logp":   base / f"input_recovery_{tag}.png",
        "plot_rank":   base / f"input_recovery_rank_{tag}.png",
        "plot_onset":  base / f"input_recovery_onset_{tag}.png",
        # CoT-only: hidden-state cache so re-runs skip the forward pass.
        "cot_hidden":  base / "cot_hidden_states.npy",
        "cot_meta":    base / "cot_hidden_states_meta.csv",
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "operand":           "#9467bd",
    "number_only":       "#17becf",
    "operand_shuffled":  "#bcbd22",
}
LABELS = {
    "operand":           "Bound operand",
    "number_only":       "Number (non-operand)",
    "operand_shuffled":  "Operand (shuffled null)",
}


def _shade_binding_band(ax, n_layers, with_label=False):
    """Shade the empirical binding band for Llama-3.3-70B (80 layers)."""
    if n_layers != 80:
        return
    ax.axvspan(
        BINDING_BAND[0], BINDING_BAND[1], alpha=0.18, color="#9467bd",
        label=f"Binding band L{BINDING_BAND[0]}–{BINDING_BAND[1]}" if with_label else None,
    )


def _safe_argmax(arr):
    if np.all(np.isnan(arr)):
        return None
    return int(np.nanargmax(arr))


def plot_logp(per_layer: np.ndarray, model_short: str, mode: str,
              out_path: Path):
    """Two-panel log-prob view: per-category curves + binding contrasts."""
    n_layers, _, _ = per_layer.shape
    means  = np.nanmean(per_layer, axis=1)
    counts = np.sum(~np.isnan(per_layer), axis=1).clip(min=1)
    sems   = np.nanstd(per_layer, axis=1) / np.sqrt(counts)
    layers = np.arange(n_layers)

    opd_i = CATEGORIES.index("operand")
    no_i  = CATEGORIES.index("number_only")
    osh_i = CATEGORIES.index("operand_shuffled")

    fig = plt.figure(figsize=(12, 8.5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)
    fig.suptitle(
        f"Operand recovery at answer position — {model_short} ({mode})",
        fontsize=13, y=0.995,
    )

    # ── Panel A: per-category log-prob ───────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    for ci, cat in enumerate(CATEGORIES):
        ax.fill_between(
            layers, means[:, ci] - 2*sems[:, ci], means[:, ci] + 2*sems[:, ci],
            color=COLORS[cat], alpha=0.18,
        )
        ax.plot(layers, means[:, ci], color=COLORS[cat], lw=2.0, label=LABELS[cat])
    _shade_binding_band(ax, n_layers, with_label=True)
    ax.set_xlim(0, n_layers - 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean log-prob")
    ax.set_title("A.  Per-category log-prob (±2 SE)", loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # ── Panel B: binding-signature contrasts ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    opd_vs_no   = means[:, opd_i] - means[:, no_i]
    opd_vs_shuf = means[:, opd_i] - means[:, osh_i]

    ax2.plot(layers, opd_vs_no, color="#9467bd", lw=2.2,
             label="log P(operand) − log P(number, non-operand)  [same-prompt baseline]")
    ax2.plot(layers, opd_vs_shuf, color="#bcbd22", lw=2.2, ls="--",
             label="log P(operand) − log P(shuffled operand)  [cross-prompt null]")
    ax2.axhline(0, color="grey", linewidth=0.8)
    _shade_binding_band(ax2, n_layers, with_label=True)
    ax2.set_xlim(0, n_layers - 1)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Log-ratio")
    ax2.set_title("B.  Binding signature (positive = operand is the preferred numeral)",
                  loc="left", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

    for curve, color in [(opd_vs_no, "#9467bd"), (opd_vs_shuf, "#bcbd22")]:
        peak_L = _safe_argmax(curve)
        if peak_L is None:
            continue
        ax2.scatter([peak_L], [curve[peak_L]], color=color, s=40, zorder=5,
                    edgecolor="black", linewidth=0.5)
        ax2.annotate(
            f"peak L{peak_L} (+{curve[peak_L]:.2f})",
            xy=(peak_L, curve[peak_L]), xytext=(peak_L + 3, curve[peak_L] + 0.3),
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Log-prob plot saved to {out_path}")


def plot_rank(rank_layer: np.ndarray, model_short: str, mode: str,
              out_path: Path, vocab_size: int = None):
    """Two-panel rank view: raw ranks (log-y) + binding-signature rank gaps."""
    n_layers, _, _ = rank_layer.shape

    means  = np.nanmean(rank_layer, axis=1)
    counts = np.sum(~np.isnan(rank_layer), axis=1).clip(min=1)
    sems   = np.nanstd(rank_layer, axis=1) / np.sqrt(counts)

    if vocab_size is None:
        vocab_size = int(np.nanmax(rank_layer)) + 1

    layers = np.arange(n_layers)
    opd_i = CATEGORIES.index("operand")
    no_i  = CATEGORIES.index("number_only")
    osh_i = CATEGORIES.index("operand_shuffled")

    fig = plt.figure(figsize=(12, 8.5))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)
    fig.suptitle(
        f"Operand recovery — rank metric ({model_short}, {mode})",
        fontsize=13, y=0.995,
    )

    # ── Panel A: raw mean rank, log-y ────────────────────────────────────────
    ax = fig.add_subplot(gs[0])
    for ci, cat in enumerate(CATEGORIES):
        ax.plot(layers, np.maximum(means[:, ci], 1.0),
                color=COLORS[cat], lw=2.0, label=LABELS[cat])
    _shade_binding_band(ax, n_layers, with_label=True)
    ax.set_yscale("log")
    ax.set_xlim(0, n_layers - 1)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Mean rank  (log scale; vocab {vocab_size})")
    ax.set_title("A.  Raw mean rank — lower = closer to top of the vocab distribution",
                 loc="left", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.4)

    # ── Panel B: rank-gap binding signature ──────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    no_vs_opd   = means[:, no_i] - means[:, opd_i]
    shuf_vs_opd = means[:, osh_i] - means[:, opd_i]

    ax2.plot(layers, no_vs_opd, color="#9467bd", lw=2.2,
             label="rank(number, non-operand) − rank(operand)  [same-prompt baseline]")
    ax2.plot(layers, shuf_vs_opd, color="#bcbd22", lw=2.2, ls="--",
             label="rank(shuffled operand) − rank(operand)  [cross-prompt null]")
    ax2.axhline(0, color="grey", linewidth=0.8)
    _shade_binding_band(ax2, n_layers, with_label=True)
    ax2.set_xlim(0, n_layers - 1)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Rank gap  (number of vocab tokens)")
    ax2.set_title("B.  Binding signature (positive = operand sits closer to the vocab head)",
                  loc="left", fontsize=11)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

    for curve, color in [(no_vs_opd, "#9467bd"), (shuf_vs_opd, "#bcbd22")]:
        peak_L = _safe_argmax(curve)
        if peak_L is None:
            continue
        ax2.scatter([peak_L], [curve[peak_L]], color=color, s=40, zorder=5,
                    edgecolor="black", linewidth=0.5)
        ax2.annotate(
            f"peak L{peak_L}\n(+{curve[peak_L]:.0f})",
            xy=(peak_L, curve[peak_L]), xytext=(peak_L + 3, curve[peak_L] * 0.7),
            fontsize=8.5, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
        )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Rank plot saved to {out_path}")


def plot_per_prompt_onset(rank_arr: np.ndarray, model_short: str, mode: str,
                          out_path: Path):
    """Histogram of per-prompt binding-crystallization layers.

    For each prompt, onset = first layer where rank(operand) < rank(shuffled
    operand) sustained for 3 consecutive layers.  Distribution width tells us
    whether binding crystallizes sharply or smears across the stack.
    """
    n_layers, n_prompts, _ = rank_arr.shape
    opd_i = CATEGORIES.index("operand")
    osh_i = CATEGORIES.index("operand_shuffled")

    onsets = []
    for p in range(n_prompts):
        r_opd = rank_arr[:, p, opd_i]
        r_shf = rank_arr[:, p, osh_i]
        if np.all(np.isnan(r_opd)) or np.all(np.isnan(r_shf)):
            continue
        mask = (r_shf - r_opd) > 0
        for L in range(0, n_layers - 2):
            if np.all(mask[L:L + 3]):
                onsets.append(L)
                break

    fig, ax = plt.subplots(figsize=(10, 4.2))
    if onsets:
        onsets_arr = np.asarray(onsets)
        ax.hist(onsets_arr, bins=np.arange(0, n_layers + 1),
                color="#9467bd", edgecolor="white", linewidth=0.4)
        med = float(np.median(onsets_arr))
        p25 = float(np.percentile(onsets_arr, 25))
        p75 = float(np.percentile(onsets_arr, 75))
        ax.axvline(med, color="black", lw=1.5, label=f"median L{med:.0f}")
        ax.axvspan(p25, p75, alpha=0.15, color="black",
                   label=f"IQR L{p25:.0f}–L{p75:.0f}")
        coverage = 100 * len(onsets) / n_prompts
        title_extra = f"  (coverage {coverage:.0f}% of prompts)"
    else:
        ax.text(0.5, 0.5, "No prompts crossed the null", ha="center", va="center",
                transform=ax.transAxes)
        title_extra = ""

    if n_layers == 80:
        ax.axvspan(BINDING_BAND[0], BINDING_BAND[1], alpha=0.20, color="#1f77b4",
                   label=f"Empirical band L{BINDING_BAND[0]}–{BINDING_BAND[1]}")
    ax.set_xlabel("Layer at which rank(operand) < rank(shuffled operand) becomes sustained")
    ax.set_ylabel("Number of prompts")
    ax.set_xlim(0, n_layers - 1)
    ax.set_title(
        f"Per-prompt binding crystallization onset — {model_short} ({mode}){title_extra}",
        fontsize=12, loc="left",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-prompt onset plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize(logp_arr: np.ndarray, rank_arr: np.ndarray,
              vocab_size: int = None) -> dict:
    """Onset metrics for binding crystallization.

    Two baselines, in increasing strictness:
      - number_only:      same-prompt non-operand numerals.
      - operand_shuffled: cross-prompt null (operands from a different prompt).

    For each baseline we report: first sustained positive contrast, first
    sustained ≥½-max, first sustained >2 SE (shuffle baseline only), peak
    layer, peak margin.  Plus per-prompt onset distribution (median + IQR)
    against the shuffle null — the population-level binding-crystallization
    layer.
    """
    opd_i = CATEGORIES.index("operand")
    no_i  = CATEGORIES.index("number_only")
    osh_i = CATEGORIES.index("operand_shuffled")

    rank_means  = np.nanmean(rank_arr, axis=1)
    rank_counts = np.sum(~np.isnan(rank_arr), axis=1).clip(min=1)
    rank_sems   = np.nanstd(rank_arr, axis=1) / np.sqrt(rank_counts)

    operand_rank = rank_means[:, opd_i]
    operand_logp = np.nanmean(logp_arr, axis=1)[:, opd_i]
    vs_numonly   = rank_means[:, no_i]  - operand_rank
    vs_shuffled  = rank_means[:, osh_i] - operand_rank
    shuffled_se  = np.sqrt(rank_sems[:, osh_i] ** 2 + rank_sems[:, opd_i] ** 2)

    def first_sustained(mask, width=3):
        if mask.size < width:
            return None
        for L in range(0, len(mask) - width + 1):
            if np.all(mask[L:L + width]):
                return int(L)
        return None

    def onset_metrics(curve, se=None):
        finite = curve[np.isfinite(curve)]
        half_max = 0.5 * float(np.nanmax(finite)) if finite.size else np.nan
        pos = first_sustained(curve > 0)
        hm  = (first_sustained(curve >= half_max)
               if np.isfinite(half_max) and half_max > 0 else None)
        s2  = (first_sustained((curve > 0) & (curve > 2 * se))
               if se is not None else None)
        if np.all(np.isnan(curve)):
            peak_L, peak_v = None, None
        else:
            peak_L = int(np.nanargmax(curve))
            peak_v = float(curve[peak_L])
        return {
            "first_sustained_positive": pos,
            "first_sustained_half_max": hm,
            "first_sustained_gt_2se":   s2,
            "peak_layer":               peak_L,
            "peak_margin":              peak_v,
        }

    summary_numonly  = onset_metrics(vs_numonly)
    summary_shuffled = onset_metrics(vs_shuffled, shuffled_se)

    # Per-prompt onset against shuffle null
    n_layers, n_prompts, _ = rank_arr.shape
    per_prompt_onset = []
    for p in range(n_prompts):
        r_opd = rank_arr[:, p, opd_i]
        r_shf = rank_arr[:, p, osh_i]
        if np.all(np.isnan(r_opd)) or np.all(np.isnan(r_shf)):
            continue
        onset = first_sustained((r_shf - r_opd) > 0)
        if onset is not None:
            per_prompt_onset.append(onset)

    if per_prompt_onset:
        arr = np.asarray(per_prompt_onset)
        onset_summary = {
            "n_prompts_with_onset": int(arr.size),
            "median": float(np.median(arr)),
            "p25":    float(np.percentile(arr, 25)),
            "p75":    float(np.percentile(arr, 75)),
            "values": arr.tolist(),
        }
    else:
        onset_summary = {
            "n_prompts_with_onset": 0,
            "median": None, "p25": None, "p75": None, "values": [],
        }

    pop_onset = onset_summary["median"]
    band_check = {
        "binding_band": list(BINDING_BAND),
        "population_median_onset": pop_onset,
        "onset_within_band": (
            pop_onset is not None and BINDING_BAND[0] <= pop_onset <= BINDING_BAND[1]
        ),
    }

    best_rank_layer = (
        int(np.nanargmin(operand_rank))
        if not np.all(np.isnan(operand_rank)) else None
    )

    return {
        "metric_description": (
            "Binding crystallization at the answer-position residual: each "
            "layer is unembedded through final norm + lm_head and the rank / "
            "logprob of bound-operand token IDs is tracked.  Binding "
            "signature = rank(shuffled operand) − rank(operand)."
        ),
        "vocab_size":                      vocab_size,
        "operand_rank_lower_is_better":    operand_rank.tolist(),
        "operand_logp":                    operand_logp.tolist(),
        "operand_vs_numonly_rank_margin":  vs_numonly.tolist(),
        "operand_vs_shuffled_rank_margin": vs_shuffled.tolist(),
        "vs_number_only":                  summary_numonly,
        "vs_shuffled_null":                summary_shuffled,
        "best_operand_rank_layer":         best_rank_layer,
        "best_operand_rank": (
            float(operand_rank[best_rank_layer]) if best_rank_layer is not None else None
        ),
        "per_prompt_onset":  onset_summary,
        "binding_band_check": band_check,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_input_paths(model_short: str,
                          dataset_name: str = "gsm_symbolic") -> tuple[Path, Path]:
    # Primary path delegates to run_template_similarity so per-dataset
    # CACHE_ROOT_OVERRIDES for large paired datasets are honored.
    from run_template_similarity import cache_path, meta_path as ts_meta_path
    primary = (
        cache_path(model_short, correct_only=False,
                   dataset_name=dataset_name, mode="direct"),
        ts_meta_path(model_short, correct_only=False,
                     dataset_name=dataset_name, mode="direct"),
    )
    if primary[0].exists() and primary[1].exists():
        return primary

    # Legacy fallbacks for the default-dataset case (older layouts).
    candidates = [primary]
    if dataset_name == "gsm_symbolic":
        candidates += [
            (
                HS_DIR / model_short / "hidden_states_all.npy",
                HS_DIR / model_short / "hidden_states_all_meta.csv",
            ),
            (
                HS_DIR / f"hidden_states_{model_short}_all.npy",
                HS_DIR / f"hidden_states_{model_short}_all_meta.csv",
            ),
        ]
    for hs_path, meta_path in candidates:
        if hs_path.exists() and meta_path.exists():
            return hs_path, meta_path
    return primary


def _build_prompt_cats(df: pd.DataFrame, tokenizer) -> tuple[list, dict, int]:
    """Build per-prompt category dicts (operand, number_only, operand_shuffled).

    Returns (prompt_cats, cat_counts, n_with_operand).
    """
    prompt_cats = []
    cat_counts = {c: 0 for c in CATEGORIES}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        operand_groups, operand_spans = operand_groups_from_binding(
            row["question"], row.get("symbol_binding", ""), tokenizer
        )
        c = {
            "operand":           operand_groups,
            "number_only":       number_only_groups(row["question"], tokenizer, operand_spans),
            "operand_shuffled":  [],
        }
        prompt_cats.append(c)
        for k, v in c.items():
            cat_counts[k] += len(v)

    rng = np.random.default_rng(0)
    n = len(prompt_cats)
    perm = rng.permutation(n)
    for i in range(n):
        if perm[i] == i and n > 1:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    for i in range(n):
        prompt_cats[i]["operand_shuffled"] = prompt_cats[int(perm[i])]["operand"]
    cat_counts["operand_shuffled"] = sum(
        len(pc["operand_shuffled"]) for pc in prompt_cats
    )

    n_with_operand = sum(1 for pc in prompt_cats if pc["operand"])
    return prompt_cats, cat_counts, n_with_operand


def _save_and_plot(
    logp_arr: np.ndarray,
    rank_arr: np.ndarray,
    cat_counts: dict,
    n_prompts: int,
    n_with_operand: int,
    vocab_size: int,
    model_short: str,
    mode: str,
    correct_only: bool = False,
):
    paths = out_paths(model_short, mode, correct_only)
    np.savez(paths["npz"], logp=logp_arr, rank=rank_arr)
    with open(paths["meta"], "w") as f:
        json.dump({
            "mode":                    mode,
            "correct_only":            correct_only,
            "categories":              CATEGORIES,
            "cat_group_counts":        cat_counts,
            "shape":                   list(logp_arr.shape),
            "n_prompts":               n_prompts,
            "n_prompts_with_operand":  n_with_operand,
            "vocab_size":              vocab_size,
            "binding_band":            list(BINDING_BAND),
            "model_short":             model_short,
        }, f, indent=2)

    summary = summarize(logp_arr, rank_arr, vocab_size)
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Binding-crystallization summary saved to {paths['summary']}")

    rank_means = np.nanmean(rank_arr, axis=1)
    print(f"\nMean rank (0 = top of vocab) per category at key layers ({mode}):")
    print(f"{'L':>4}  " + "  ".join(f"{c:>18}" for c in CATEGORIES) +
          f"  {'no − opd':>9}  {'shuf − opd':>11}")
    for L in [0, 10, 20, 25, 30, 35, 38, 39, 40, 50, 70, 79]:
        if L >= rank_arr.shape[0]:
            continue
        no_minus = rank_means[L, CATEGORIES.index("number_only")] \
                   - rank_means[L, CATEGORIES.index("operand")]
        sh_minus = rank_means[L, CATEGORIES.index("operand_shuffled")] \
                   - rank_means[L, CATEGORIES.index("operand")]
        print(f"{L:>4}  " + "  ".join(f"{rank_means[L,ci]:>18.0f}"
                                      for ci in range(len(CATEGORIES))) +
              f"  {no_minus:>+9.0f}  {sh_minus:>+11.0f}")

    print(f"\nBinding crystallization onsets ({mode}):")
    for key in ("vs_number_only", "vs_shuffled_null"):
        d = summary[key]
        if d["peak_margin"] is None:
            print(f"  {key:>16}: no signal")
            continue
        print(
            f"  {key:>16}: first+ L{d['first_sustained_positive']}, "
            f"half-max L{d['first_sustained_half_max']}, "
            f"peak L{d['peak_layer']} (+{d['peak_margin']:.0f} ranks)"
        )
    pp = summary["per_prompt_onset"]
    print(
        f"  per-prompt onset (shuffled null, width-3): median L{pp['median']} "
        f"(IQR L{pp['p25']}–L{pp['p75']}, n={pp['n_prompts_with_onset']})"
    )
    bbc = summary["binding_band_check"]
    print(
        f"  binding band check: median onset L{bbc['population_median_onset']} "
        f"vs band L{bbc['binding_band'][0]}–L{bbc['binding_band'][1]} "
        f"→ within band = {bbc['onset_within_band']}"
    )

    plot_logp(logp_arr, model_short, mode, paths["plot_logp"])
    plot_rank(rank_arr, model_short, mode, paths["plot_rank"], vocab_size=vocab_size)
    plot_per_prompt_onset(rank_arr, model_short, mode, paths["plot_onset"])


def _load_or_extract_cot_hidden(
    df: pd.DataFrame, model, tokenizer, paths: dict, batch_size: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load CoT hidden states from disk if present and aligned, otherwise extract
    and cache them.  Returns (hidden_states, df_aligned)."""
    cot_meta_path = paths["cot_meta"]
    cot_hs_path   = paths["cot_hidden"]

    if cot_hs_path.exists() and cot_meta_path.exists():
        cached_meta = pd.read_csv(cot_meta_path)
        # Sanity check: same (original_id, instance) set, same order
        expected_meta = cached_meta[["original_id", "instance"]]
        expected_fingerprint = _cot_cache_fingerprint(tokenizer)
        cache_version_ok = (
            "cache_version" in cached_meta.columns
            and (cached_meta["cache_version"] == COT_CACHE_VERSION).all()
        )
        max_length_ok = (
            "max_length" in cached_meta.columns
            and (cached_meta["max_length"] == COT_MAX_LENGTH).all()
        )
        fingerprint_ok = (
            "prompt_fingerprint" in cached_meta.columns
            and (cached_meta["prompt_fingerprint"] == expected_fingerprint).all()
        )
        if (
            len(cached_meta) > 0
            and cache_version_ok
            and max_length_ok
            and fingerprint_ok
            and expected_meta.equals(expected_meta.drop_duplicates().reset_index(drop=True))
        ):
            print(f"Loading cached CoT hidden states: {cot_hs_path}")
            hidden = np.load(cot_hs_path)
            df_aligned = expected_meta.merge(
                df,
                on=["original_id", "instance"],
                how="left",
                validate="one_to_one",
            ).reset_index(drop=True)
            if (
                hidden.ndim == 3
                and hidden.shape[0] == len(cached_meta)
                and df_aligned["question"].notna().all()
            ):
                print(f"  shape: {hidden.shape}")
                return hidden, df_aligned
            print(
                f"Cached CoT hidden states at {cot_hs_path} have shape "
                f"{hidden.shape}, but meta has {len(cached_meta)} rows; re-extracting."
            )
        else:
            print(
                f"Cached CoT meta at {cot_meta_path} does not align with current df; "
                "re-extracting."
            )

    print("Extracting CoT hidden states (this is a fresh forward pass)…")
    hidden, kept_idx = extract_cot_hidden_states(
        df, model, tokenizer, batch_size=batch_size,
    )
    df_aligned = df.iloc[kept_idx].reset_index(drop=True)

    np.save(cot_hs_path, hidden)
    cot_meta = df_aligned[["original_id", "instance"]].copy()
    cot_meta["cache_version"] = COT_CACHE_VERSION
    cot_meta["max_length"] = COT_MAX_LENGTH
    cot_meta["prompt_fingerprint"] = _cot_cache_fingerprint(tokenizer)
    cot_meta.to_csv(cot_meta_path, index=False)
    print(f"Cached CoT hidden states → {cot_hs_path}")
    return hidden, df_aligned


def _apply_correct_only(
    df: pd.DataFrame, hidden: np.ndarray, mode: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter df + hidden state rows to model-correct instances only."""
    col = "original_direct_correctness" if mode == "direct" else "original_cot_correctness"
    if col not in df.columns:
        raise ValueError(
            f"--correct_only requested but column '{col}' not in gsm_symbolic CSVs."
        )
    mask = _coerce_bool(df[col]).fillna(False).to_numpy(dtype=bool)
    before = len(df)
    df_filt = df.loc[mask].reset_index(drop=True)
    hidden_filt = hidden[mask]
    print(f"  correct_only filter ({col}): kept {len(df_filt)} / {before}")
    return df_filt, hidden_filt


def run(model_id: str, mode: str, batch_size: int, extract_batch_size: int,
        correct_only: bool = False, dataset_name: str = "gsm_symbolic"):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    paths = out_paths(model_short, mode, correct_only)

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=CACHE_DIR, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {model_id}…")
    model, _ = load_model(model_id)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    if mode == "direct":
        hs_path, meta_path = _resolve_input_paths(model_short, dataset_name)
        if not hs_path.exists():
            raise FileNotFoundError(
                f"Cached hidden states not found at {hs_path}. "
                "Run run_template_similarity.py first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Hidden-state metadata CSV not found at {meta_path}."
            )
        print(f"Loading direct-mode hidden states: {hs_path}")
        hidden = np.load(hs_path)
        print(f"  shape: {hidden.shape}")
        df = load_aligned_questions(model_id, meta_path)
        print(f"Aligned {len(df)} questions")
        if correct_only:
            df, hidden = _apply_correct_only(df, hidden, mode)
    elif mode == "cot":
        df_all = load_gsm_symbolic_df(model_id)
        if "original_cot" not in df_all.columns:
            raise ValueError(
                "gsm_symbolic CSVs are missing the `original_cot` column — "
                "this model has no cached CoT responses to read."
            )
        hidden, df = _load_or_extract_cot_hidden(
            df_all, model, tokenizer, paths, batch_size=extract_batch_size,
        )
        print(f"Aligned {len(df)} CoT prompts")
        if correct_only:
            df, hidden = _apply_correct_only(df, hidden, mode)
    else:
        raise ValueError(f"Unknown --mode: {mode}")

    print("Building operand and number_only groups…")
    prompt_cats, cat_counts, n_with_operand = _build_prompt_cats(df, tokenizer)
    n = len(prompt_cats)
    print(f"  Total unique groups per category: {cat_counts}")
    print(
        f"  Prompts with ≥1 bound operand: {n_with_operand} / {n} "
        f"({100*n_with_operand/max(n,1):.1f}%)"
    )

    logp_arr, rank_arr = decode_per_layer(
        hidden, model, prompt_cats, batch_size=batch_size,
    )

    vocab_size = int(
        getattr(model.config, "vocab_size", None)
        or model.lm_head.weight.shape[0]
    )

    _save_and_plot(
        logp_arr, rank_arr, cat_counts, n, n_with_operand,
        vocab_size, model_short, mode, correct_only=correct_only,
    )


def plot_only(model_id: str, mode: str, correct_only: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    paths = out_paths(model_short, mode, correct_only)

    if not paths["npz"].exists():
        raise FileNotFoundError(
            f"No results at {paths['npz']}.  Run without --plot_only first."
        )
    arr = np.load(paths["npz"])
    logp_arr = arr["logp"]
    if "rank" not in arr.files:
        raise ValueError(
            f"Cached {paths['npz']} is missing the rank array.  "
            "Rerun without --plot_only to regenerate."
        )
    rank_arr = arr["rank"]

    if logp_arr.shape[-1] != len(CATEGORIES) or rank_arr.shape[-1] != len(CATEGORIES):
        raise ValueError(
            f"Cached arrays have {logp_arr.shape[-1]} categories; this script "
            f"expects {len(CATEGORIES)} ({CATEGORIES}).  Rerun without "
            "--plot_only to regenerate."
        )

    vocab_size = None
    if paths["meta"].exists():
        with open(paths["meta"]) as f:
            vocab_size = json.load(f).get("vocab_size")

    summary = summarize(logp_arr, rank_arr, vocab_size)
    with open(paths["summary"], "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Binding-crystallization summary saved to {paths['summary']}")

    plot_logp(logp_arr, model_short, mode, paths["plot_logp"])
    plot_rank(rank_arr, model_short, mode, paths["plot_rank"], vocab_size=vocab_size)
    plot_per_prompt_onset(rank_arr, model_short, mode, paths["plot_onset"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--mode", choices=["direct", "cot"], default="direct",
                        help="direct: reuse cached residuals from "
                             "run_template_similarity.py.  cot: extract fresh "
                             "residuals after the cached CoT reasoning chain.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for the per-layer unembed decode.")
    parser.add_argument("--extract_batch_size", type=int, default=2,
                        help="Batch size for the CoT forward pass (long context).")
    parser.add_argument("--correct_only", action="store_true",
                        help="Restrict to instances the model answered correctly "
                             "(filters by original_direct_correctness for direct, "
                             "original_cot_correctness for cot).")
    parser.add_argument("--dataset", default="gsm_symbolic",
                        help="Dataset name for the hidden-state cache lookup.")
    parser.add_argument("--plot_only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_only(args.model_id, args.mode, args.correct_only)
    else:
        run(args.model_id, args.mode, args.batch_size, args.extract_batch_size,
            correct_only=args.correct_only, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
