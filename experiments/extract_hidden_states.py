"""
Extract and save hidden states for interpretability experiments.

For each question, runs a forward pass and saves the hidden state at the last
token of the question for every layer.

Output: numpy array of shape (N, n_layers+1, hidden_size) saved to:
    results/hidden_states/{dataset}_hidden_{model}.npy

Usage:
    python extract_hidden_states.py --dataset gsm8k        --model_name meta-llama/Llama-3.3-70B-Instruct
    python extract_hidden_states.py --dataset gsm_symbolic  --model_name meta-llama/Llama-3.3-70B-Instruct
    python extract_hidden_states.py --dataset gsm_noop      --model_name meta-llama/Llama-3.3-70B-Instruct
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import CACHE_DIR
from instructions import TASK_CONFIG
from utils.data_utils import (
    hidden_states_path,
    load_gsm8k,
    load_gsm_noop,
    load_gsm_symbolic,
    probe_dir,
    sym_noop_idx_path,
)

# Use the original CoT instruction — matches the inference runs we're analysing
_INSTRUCTION = TASK_CONFIG["original"]["cot"]

DATASETS = ["gsm8k", "gsm_symbolic", "gsm_noop"]


def extract(df, question_col, model_name, output_path, batch_size):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=CACHE_DIR, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    questions = df[question_col].tolist()
    all_hidden = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Extracting"):
        batch_qs = questions[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "user",   "content": f"{q}\n{_INSTRUCTION}"},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_qs
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            out = model(**enc)

        for b in range(len(batch_qs)):
            # Safely get the last valid token index regardless of padding side
            last = enc["attention_mask"][b].nonzero()[-1].item()
            states = torch.stack(
                [layer[b, last, :].cpu() for layer in out.hidden_states], dim=0
            ).float().numpy()   # (n_layers+1, H)
            all_hidden.append(states)

    arr = np.stack(all_hidden, axis=0)  # (N, n_layers+1, H)
    print(f"Shape: {arr.shape}")
    np.save(output_path, arr)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model_name", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--probe_dir", default=None)
    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]
    base_dir = probe_dir(model_short, args.probe_dir)
    output_path = hidden_states_path(args.dataset, model_short, base_dir)

    if os.path.exists(output_path):
        print(f"Already exists: {output_path}. Delete to re-extract.")
        return

    if args.dataset == "gsm8k":
        # Exclude the 100 NoOp templates to avoid contamination in probe training
        df = load_gsm8k(model_short, exclude_noop_templates=True)
    elif args.dataset == "gsm_symbolic":
        # For divergence: align gsm_symbolic rows one-to-one with gsm_noop by (id, instance)
        # so gsm_symbolic_hidden and gsm_noop_hidden are row-aligned for direct comparison.
        sym_df   = load_gsm_symbolic(model_short)
        noop_df  = load_gsm_noop(model_short)
        noop_keys = noop_df[["id", "instance"]].reset_index().rename(columns={"index": "noop_idx"})
        sym_keys  = sym_df[["id", "instance"]].reset_index().rename(columns={"index": "sym_idx"})
        matched = noop_keys.merge(sym_keys, on=["id", "instance"])
        df = sym_df.iloc[matched["sym_idx"].values].reset_index(drop=True)
        # Save the noop row indices so downstream code can slice noop_df to the same rows
        noop_idx_path = sym_noop_idx_path(model_short, base_dir)
        np.save(noop_idx_path, matched["noop_idx"].values)
        print(f"Aligned {len(df)} gsm_symbolic instances to gsm_noop (one-to-one by id+instance)")
        print(f"Saved noop alignment indices to {noop_idx_path}")
    else:  # gsm_noop
        df = load_gsm_noop(model_short)

    print(f"Loaded {len(df)} instances from {args.dataset}")

    extract(
        df=df,
        question_col="question",
        model_name=args.model_name,
        output_path=output_path,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
