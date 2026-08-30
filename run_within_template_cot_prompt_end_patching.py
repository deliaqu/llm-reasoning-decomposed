"""
Within-template prompt_end activation patching for variable-binding analysis.

Mirrors run_direct_swap_activation_patching.py's prompt_end design but with:
- Within-template pairs (same GSM-Symbolic template, different operand bindings)
  instead of p1-vs-symbolic prompts.
- CoT-style prompts: chat_template(question + COT_INSTRUCTION) + own_CoT_prefix + "#### ".
- Each prompt's own cached CoT trace is appended (truncated before its '####').
- Patch position: prompt_end (the trailing space after '#### ', identical to
  ROME/IOI convention used in direct-mode prompt_end).
- Readout: log P(receiver's own first answer token) at the last position.

Hypothesis we are testing: in cot_boundary patching, the smearing of single-
component (attn_output, mlp) effects comes from free CoT generation between
patch and readout, not from CoT prompt context. By using a fixed cached CoT
trace and reading immediately at the answer position, the downstream distance
between patch and readout collapses — we expect to recover an attn_output spike
at L=40 similar to direct mode.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import ACTIVATION_PATCHING_RESULT_DIR, ANSWER_PREFIX, COT_INSTRUCTION, MODEL_NAME_MAP
from run_cot_swap_activation_patching import (
    encode_prompt, layer_module, make_cross_position_patching_hook, make_save_hook,
)
from run_cot_swap_logit_lens import (
    _bootstrap_ci, _style_layer_axis, _template_means, cot_prefix_before_hash,
)
from run_noop_activation_patching import build_gsm_sym_within_template_cot_pairs
from utils.noop_utils import load_model, to_serializable
from utils.shared_utils import remove_all_hooks

OUT_DIR = Path(ACTIVATION_PATCHING_RESULT_DIR).parent / "gsm_sym_within_template_patching"
MODE = "cot_boundary"
PATCH_POSITIONS = "prompt_end"
STAGE_BOUNDARIES = [
    (22, "Abstraction"),
    (36, "Symbolic\nformulation"),
    (40, "Variable\nbinding"),
]


def output_path(model_name, scope, layers=None):
    layers_tag = ("_l" + "-".join(map(str, layers))) if layers else ""
    return OUT_DIR / model_name / MODE / PATCH_POSITIONS / scope / f"source_to_target{layers_tag}.jsonl"


def build_prompt(question, cot_prefix, tokenizer):
    """chat_template(question + COT_INSTRUCTION) + cot_prefix + '#### '."""
    messages = [{"role": "user", "content": f"{question}\n{COT_INSTRUCTION}"}]
    base = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return base + cot_prefix + ANSWER_PREFIX


def token_logprob(model, inputs, token_id):
    with torch.no_grad():
        out = model(**inputs)
        lp = F.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    return lp[0, token_id].item()


def patch_one(model, tokenizer, row, scope, layers, min_destruction_gap):
    """Patch source prompt_end activations into target prompt_end; measure
    destruction of receiver's (target's) own first answer token."""
    device = next(model.parameters()).device
    cot_c = cot_prefix_before_hash(row["original_cot_c"])
    cot_w = cot_prefix_before_hash(row["original_cot_w"])
    if cot_c is None or cot_w is None:
        return {"skipped": "no_cot"}

    prompt_c = build_prompt(row["question_c"], cot_c, tokenizer)
    prompt_w = build_prompt(row["question_w"], cot_w, tokenizer)
    inputs_c = encode_prompt(prompt_c, tokenizer, device)
    inputs_w = encode_prompt(prompt_w, tokenizer, device)

    tok_c = tokenizer.encode(str(row["answer_c"]).strip(), add_special_tokens=False)[0]
    tok_w = tokenizer.encode(str(row["answer_w"]).strip(), add_special_tokens=False)[0]
    if tok_c == tok_w:
        return {"skipped": "same_token"}

    # Direction: source=c → target=w. Receiver=w, readout=tok_w.
    recv_clean_lp = token_logprob(model, inputs_w, tok_w)
    # Cross-prompt floor: log P(tok_w | source prompt).
    recv_floor_lp = token_logprob(model, inputs_c, tok_w)
    destruction_denom = recv_clean_lp - recv_floor_lp
    if destruction_denom < min_destruction_gap:
        return {"skipped": "small_gap"}

    # prompt_end: last token of each prompt.
    donor_idx = [inputs_c["input_ids"].shape[1] - 1]
    recv_idx = [inputs_w["input_ids"].shape[1] - 1]

    num_layers = model.config.num_hidden_layers
    active_layers = layers if layers else list(range(num_layers))

    remove_all_hooks(model)
    saved = {}
    handles = []
    for L in active_layers:
        module, _ = layer_module(model, scope, L)
        handles.append(module.register_forward_hook(make_save_hook(saved, L)))
    with torch.no_grad():
        model(**inputs_c)
    for h in handles:
        h.remove()

    patched_lps = np.full(num_layers, np.nan)
    destruction = np.full(num_layers, np.nan)
    for L in active_layers:
        module, _ = layer_module(model, scope, L)
        ph = module.register_forward_hook(
            make_cross_position_patching_hook(saved[L], donor_idx, recv_idx)
        )
        lp = token_logprob(model, inputs_w, tok_w)
        ph.remove()
        patched_lps[L] = lp
        destruction[L] = (recv_clean_lp - lp) / destruction_denom

    remove_all_hooks(model)

    return {
        "source_tok_len": int(inputs_c["input_ids"].shape[1]),
        "target_tok_len": int(inputs_w["input_ids"].shape[1]),
        "source_first_answer_token_id": int(tok_c),
        "target_first_answer_token_id": int(tok_w),
        "recv_clean_logprob": recv_clean_lp,
        "recv_floor_logprob": recv_floor_lp,
        "destruction_denom": destruction_denom,
        "patched_logprobs": patched_lps,
        "destruction": destruction,
    }


def run(model, tokenizer, model_id, model_name, scope, layers=None,
        max_pairs_per_orig=None, min_destruction_gap=0.5, overwrite=False):
    pairs = build_gsm_sym_within_template_cot_pairs(
        model_id, max_pairs_per_orig=max_pairs_per_orig
    )
    out_path = output_path(model_name, scope, layers)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and out_path.exists():
        out_path.unlink()

    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                done.add((r["original_id"], r["instance_c"], r["instance_w"]))
        print(f"Resuming - {len(done)} pairs done")

    print(
        f"Built {len(pairs)} within-template pairs across "
        f"{pairs['original_id'].nunique()} templates"
    )
    model.eval()
    skips = {"no_cot": 0, "same_token": 0, "small_gap": 0}
    with open(out_path, "a") as f_out:
        for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc=scope):
            key = (int(row["original_id"]), int(row["instance_c"]), int(row["instance_w"]))
            if key in done:
                continue
            metrics = patch_one(model, tokenizer, row, scope, layers, min_destruction_gap)
            tag = metrics.get("skipped")
            if tag:
                skips[tag] += 1
                continue
            result = {
                "original_id": int(row["original_id"]),
                "instance_c": int(row["instance_c"]),
                "instance_w": int(row["instance_w"]),
                "answer_c": str(row["answer_c"]),
                "answer_w": str(row["answer_w"]),
                **{k: to_serializable(v) for k, v in metrics.items()},
            }
            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

    for tag, n in skips.items():
        if n:
            print(f"Skipped {n} pairs: {tag}")
    print(f"Results -> {out_path}")


def plot(model_name, scope, min_destruction_gap=0.5, layers=None):
    path = output_path(model_name, scope, layers=layers)
    if not path.exists():
        print(f"Missing: {path}")
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows
            if float(r.get("destruction_denom", 0.0)) >= min_destruction_gap]
    if not rows:
        print(f"No rows survive min_destruction_gap={min_destruction_gap}")
        return
    dest = np.array([r["destruction"] for r in rows], dtype=float)
    cids = np.array([r["original_id"] for r in rows])
    finite = np.isfinite(dest).any(axis=0)
    if not finite.any():
        return
    dest = dest[:, finite]
    layers_arr = np.arange(finite.shape[0])[finite] + 1
    num_layers = int(finite.shape[0])

    tids, dt = _template_means(np.nan_to_num(dest), cids)
    lo, hi = _bootstrap_ci(dt, n_boot=1000, ci=0.95)
    mean = dt.mean(0)

    color = {"layer": "#2166ac", "attn_output": "#1b7837", "mlp": "#d6604d"}.get(scope, "#333333")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(layers_arr, mean, color=color, lw=1.8,
            label=f"source→target prompt_end (scope={scope}, T={len(tids)})")
    ax.fill_between(layers_arr, lo, hi, color=color, alpha=0.18)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.axhline(1, color="black", lw=0.8, alpha=0.3)
    ax.set_ylabel("Normalized destruction of\nreceiver answer")
    _style_layer_axis(ax, num_layers, stage_boundaries=STAGE_BOUNDARIES)
    ax.legend(fontsize=9.5, loc="upper left", frameon=True)
    plt.tight_layout()
    save = path.with_suffix(".png")
    plt.savefig(save, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {save} (N={len(rows)}, T={len(tids)})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_id", default="meta-llama/Llama-3.3-70B-Instruct",
        choices=list(MODEL_NAME_MAP.keys()),
    )
    p.add_argument("--scope", choices=["layer", "mlp", "attn_output"], required=True)
    p.add_argument("--layers", type=int, nargs="*", default=None)
    p.add_argument("--max_pairs_per_orig", type=int, default=None)
    p.add_argument("--min_destruction_gap", type=float, default=0.5)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--plot_only", action="store_true")
    args = p.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]
    if not args.plot_only:
        model, tokenizer = load_model(args.model_id)
        run(model, tokenizer, args.model_id, model_name, args.scope,
            layers=args.layers, max_pairs_per_orig=args.max_pairs_per_orig,
            min_destruction_gap=args.min_destruction_gap, overwrite=args.overwrite)
    # A layer-sharded JSONL has only one finite layer per row — a per-shard plot
    # is degenerate. Skip; merge shards into the unsharded file before plotting.
    if args.layers:
        print(
            f"Layer-sharded run for layers={args.layers}; skipping plot. "
            f"Merge shards into source_to_target.jsonl, then run with --plot_only."
        )
    else:
        plot(model_name, args.scope, args.min_destruction_gap)
