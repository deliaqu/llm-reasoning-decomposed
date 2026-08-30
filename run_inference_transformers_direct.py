from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from instructions import TASK_CONFIG
from paths import CACHE_DIR, DATA_DIR, RESULT_DIR
from run_evaluation import (
    evaluate_final_answer,
    evaluate_final_answer_text,
    postprocess_generation,
)

# Datasets whose gold answers are text (names, dates, ...) rather than
# integers: use the original_text instructions + text-aware correctness.
TEXT_ANSWER_DATASETS = {"phantomwiki", "phantomwiki_roles"}


GSM_SYMBOLIC_PADDED_LT2 = "gsm_symbolic_padded_to_p1_len_delta_abs_lt2"
GSM_SYMBOLIC_PADDED_DELTA0 = "gsm_symbolic_padded_to_p1_len_delta_0"
GSM_SYMBOLIC_TO_P1 = "gsm_symbolic_to_p1"
GSM_PADDED_SYMBOLIC = "gsm_padded_symbolic"
GSM_PADDED_SYMBOLIC_P1_ALIGNED = "gsm_padded_symbolic_p1_aligned"
GSM_P1_FROM_SYMBOLIC = "gsm_p1_from_symbolic"
GSM_P1_FROM_SYMBOLIC_STRICT = "gsm_p1_from_symbolic_strict_valid"
GSM_SYMBOLIC_PAIRS = "gsm_symbolic_pairs"
GSM_SYMBOLIC_SPLIT_DATASETS = {
    GSM_SYMBOLIC_PADDED_LT2: {
        "split_dir": "split_datasets_symbolic_padded_to_p1_len_delta_abs_lt2",
        "file_stem": "test_gsm_symbolic_padded_to_p1",
    },
    GSM_SYMBOLIC_PADDED_DELTA0: {
        "split_dir": "split_datasets_symbolic_padded_to_p1_len_delta_0",
        "file_stem": "test_gsm_symbolic_padded_to_p1",
    },
    GSM_SYMBOLIC_TO_P1: {
        "split_dir": "split_datasets_symbolic_to_p1",
        "file_stem": "test_gsm_symbolic_to_p1",
    },
}


def load_model_transformers(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=CACHE_DIR,
        dtype=torch.float16,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer


def resolve_data_path(dataset_name: str, set_id: int | None):
    if set_id is not None:
        if dataset_name == "gsm_p1":
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_p1/test_gsm_p1_set_{set_id}.csv",
                f"gsm_p1_set_{set_id}",
            )
        if dataset_name == GSM_P1_FROM_SYMBOLIC:
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_p1_from_symbolic/test_gsm_p1_from_symbolic_set_{set_id}.csv",
                f"{GSM_P1_FROM_SYMBOLIC}_set_{set_id}",
            )
        if dataset_name == GSM_P1_FROM_SYMBOLIC_STRICT:
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_p1_from_symbolic_strict_valid/test_gsm_p1_from_symbolic_strict_valid_set_{set_id}.csv",
                f"{GSM_P1_FROM_SYMBOLIC_STRICT}_set_{set_id}",
            )
        if dataset_name == "gsm_p2":
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_p2/test_gsm_p2_set_{set_id}.csv",
                f"gsm_p2_set_{set_id}",
            )
        if dataset_name == "gsm_noop":
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_noop/test_gsm_noop_set_{set_id}.csv",
                f"gsm_noop_set_{set_id}",
            )
        if dataset_name == "gsm_noop_clean":
            return (
                f"{DATA_DIR}/gsm_symbolic/clean_noop/split_datasets_noop/test_gsm_noop_set_{set_id}.csv",
                f"gsm_noop_clean_set_{set_id}",
            )
        if dataset_name == "gsm_noop_clean_hard":
            return (
                f"{DATA_DIR}/gsm_symbolic/clean_noop/split_datasets_noop_hard/test_gsm_noop_set_{set_id}.csv",
                f"gsm_noop_clean_hard_set_{set_id}",
            )
        if dataset_name == "gsm_filler":
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_filler/test_gsm_filler_set_{set_id}.csv",
                f"gsm_filler_set_{set_id}",
            )
        if dataset_name == "gsm_filler_df":
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_filler_df/test_gsm_filler_df_set_{set_id}.csv",
                f"gsm_filler_df_set_{set_id}",
            )
        if dataset_name == "gsm_filler_hard":
            return (
                f"{DATA_DIR}/gsm_symbolic/clean_noop/split_datasets_filler_hard/test_gsm_filler_set_{set_id}.csv",
                f"gsm_filler_hard_set_{set_id}",
            )
        if dataset_name == GSM_PADDED_SYMBOLIC:
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_padded_symbolic/test_gsm_padded_symbolic_set_{set_id}.csv",
                f"{GSM_PADDED_SYMBOLIC}_set_{set_id}",
            )
        if dataset_name == GSM_PADDED_SYMBOLIC_P1_ALIGNED:
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_padded_symbolic_p1_aligned/test_gsm_padded_symbolic_p1_aligned_set_{set_id}.csv",
                f"{GSM_PADDED_SYMBOLIC_P1_ALIGNED}_set_{set_id}",
            )
        if dataset_name == GSM_SYMBOLIC_PAIRS:
            return (
                f"{DATA_DIR}/gsm_symbolic/split_datasets_pairs/test_gsm_symbolic_pairs_set_{set_id}.csv",
                f"gsm_symbolic_pairs_set_{set_id}",
            )
        if dataset_name in GSM_SYMBOLIC_SPLIT_DATASETS:
            dataset_config = GSM_SYMBOLIC_SPLIT_DATASETS[dataset_name]
            return (
                f"{DATA_DIR}/gsm_symbolic/{dataset_config['split_dir']}/{dataset_config['file_stem']}_set_{set_id}.csv",
                f"{dataset_name}_set_{set_id}",
            )
        return (
            f"{DATA_DIR}/gsm_symbolic/split_datasets/test_gsm_symbolic_set_{set_id}.csv",
            f"gsm_symbolic_set_{set_id}",
        )

    if dataset_name in [
        "gsm_p1",
        "gsm_p2",
        "gsm_symbolic",
        GSM_SYMBOLIC_PADDED_LT2,
        GSM_SYMBOLIC_PADDED_DELTA0,
        GSM_SYMBOLIC_TO_P1,
        GSM_PADDED_SYMBOLIC,
        GSM_PADDED_SYMBOLIC_P1_ALIGNED,
        GSM_P1_FROM_SYMBOLIC,
        GSM_P1_FROM_SYMBOLIC_STRICT,
        "gsm_noop",
        "gsm_filler",
        "gsm_filler_df",
    ]:
        return f"{DATA_DIR}/gsm_symbolic/test_{dataset_name}.csv", dataset_name
    if dataset_name == "gsm_noop_clean":
        return f"{DATA_DIR}/gsm_symbolic/clean_noop/test_gsm_noop.csv", dataset_name
    if dataset_name == "gsm_noop_clean_hard":
        return f"{DATA_DIR}/gsm_symbolic/clean_noop/test_gsm_noop_hard.csv", dataset_name
    if dataset_name == "gsm_filler_hard":
        return f"{DATA_DIR}/gsm_symbolic/clean_noop/test_gsm_filler_hard.csv", dataset_name
    return f"{DATA_DIR}/test_{dataset_name}.csv", dataset_name


def resolve_output_path(dataset_tag: str, model_short_name: str):
    target_dir = os.path.join(RESULT_DIR, "transformers_direct")
    # Check the more specific gsm_symbolic_pairs prefix BEFORE the generic
    # gsm_symbolic prefix so it doesn't get shadowed.
    if dataset_tag.startswith("gsm_symbolic_pairs"):
        target_dir = os.path.join(target_dir, "gsm_symbolic_pairs")
    elif dataset_tag.startswith("gsm_padded_symbolic_p1_aligned"):
        # Check the more-specific prefix BEFORE the generic gsm_padded_symbolic
        # prefix so it isn't shadowed.
        target_dir = os.path.join(target_dir, "gsm_padded_symbolic_p1_aligned")
    elif dataset_tag.startswith("gsm_padded_symbolic"):
        target_dir = os.path.join(target_dir, "gsm_padded_symbolic")
    elif dataset_tag.startswith("gsm_symbolic"):
        target_dir = os.path.join(target_dir, "gsm_symbolic")
    elif dataset_tag.startswith("gsm_p1"):
        target_dir = os.path.join(target_dir, "gsm_p1")
    elif dataset_tag.startswith("gsm_p2"):
        target_dir = os.path.join(target_dir, "gsm_p2")
    elif dataset_tag.startswith("gsm_noop_clean_hard"):
        target_dir = os.path.join(target_dir, "gsm_noop_clean_hard")
    elif dataset_tag.startswith("gsm_noop_clean"):
        target_dir = os.path.join(target_dir, "gsm_noop_clean")
    elif dataset_tag.startswith("gsm_noop"):
        target_dir = os.path.join(target_dir, "gsm_noop")
    elif dataset_tag.startswith("gsm_filler_hard"):
        target_dir = os.path.join(target_dir, "gsm_filler_hard")
    elif dataset_tag.startswith("gsm_filler_df"):
        target_dir = os.path.join(target_dir, "gsm_filler_df")
    elif dataset_tag.startswith("gsm_filler"):
        target_dir = os.path.join(target_dir, "gsm_filler")
    os.makedirs(target_dir, exist_ok=True)
    return os.path.join(target_dir, f"{dataset_tag}_results_{model_short_name}.csv")


def generate_original(
    df: pd.DataFrame,
    model,
    tokenizer,
    batch_size: int,
    instruction_key: str,
    output_column: str,
    max_new_tokens: int,
    task_cfg_key: str = "original",
    capture_hidden: bool = False,
    hs_chunk: int = 2,
):
    """Generate answers; when capture_hidden, ALSO return the per-layer
    residuals at the answer-prefix position (last prompt token) for every row.

    ONE-PASS DESIGN CONTRACT: inference and residual extraction happen in the
    SAME job — probe pipelines must consume the cache this writes, never
    re-read the dataset with a second GPU pass. The capture is a prompt-only
    forward (output_hidden_states) run in chunks of `hs_chunk`, decoupled
    from the generation batch because hidden states for all layers/positions
    are memory-heavy on long prompts (~81 x seq x 8192 fp16 per row).
    """
    cfg = TASK_CONFIG[task_cfg_key]
    instruction = cfg[instruction_key]
    prompts = []
    for question in df["question"].tolist():
        messages = [{"role": "user", "content": f"{question}\n{instruction}"}]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    outputs = []
    hidden_chunks = [] if capture_hidden else None
    eos_token_id = model.config.eos_token_id
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        end = min(start + batch_size, len(prompts))
        print(f"Generating {output_column} rows {start}-{end - 1} / {len(prompts) - 1}")
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        for i in range(len(batch_prompts)):
            continuation_ids = generated[i, prompt_len:].detach().cpu().tolist()
            text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
            outputs.append(text)

        if capture_hidden:
            # prompt-only forward per sub-chunk; left padding => position -1
            # is the answer prefix for every row
            for cs in range(0, len(batch_prompts), hs_chunk):
                enc = tokenizer(
                    batch_prompts[cs:cs + hs_chunk],
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                ).to(model.device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                last = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1)
                hidden_chunks.append(last.to(torch.float16).cpu().numpy())
                del out

    if capture_hidden:
        import numpy as np
        return outputs, np.concatenate(hidden_chunks, axis=0)
    return outputs, None


def hidden_states_path(dataset_tag: str, model_short_name: str,
                       num_shards: int = 1, shard_index: int = 0) -> str:
    """Canonical location of the one-pass answer-prefix residual cache."""
    d = os.path.join(CACHE_DIR, "prompt_hidden_states")
    os.makedirs(d, exist_ok=True)
    stem = f"{dataset_tag}_{model_short_name}_direct"
    if num_shards > 1:
        stem += f".shard{shard_index}of{num_shards}"
    return os.path.join(d, f"{stem}.npz")


def column_missing_or_incomplete(df: pd.DataFrame, column: str) -> bool:
    return column not in df.columns or df[column].isna().any()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=[
            "gsm8k",
            "svamp",
            "svamp_variants",
            "phantomwiki",
            "phantomwiki_roles",
            "gsm_symbolic",
            GSM_SYMBOLIC_PADDED_LT2,
            GSM_SYMBOLIC_PADDED_DELTA0,
            GSM_SYMBOLIC_TO_P1,
            GSM_P1_FROM_SYMBOLIC,
            GSM_P1_FROM_SYMBOLIC_STRICT,
            "gsm_p1",
            "gsm_p2",
            "gsm_noop",
            "gsm_noop_clean",
            "gsm_noop_clean_hard",
            "gsm_filler",
            "gsm_filler_df",
            "gsm_filler_hard",
            GSM_PADDED_SYMBOLIC,
            GSM_PADDED_SYMBOLIC_P1_ALIGNED,
            GSM_SYMBOLIC_PAIRS,
        ],
        required=True,
    )
    parser.add_argument("--set_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    # Data-parallel sharding: split the dataset into --num_shards contiguous
    # slices and process only --shard_index here. Defaults (1, 0) => no
    # sharding, so existing single-GPU callers are unaffected. Each shard
    # writes a shard-suffixed CSV; merge with merge_shard_results.py.
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["original_direct", "original_cot"],
        default=["original_direct", "original_cot"],
        help="Which original-answer prompts to run.",
    )
    parser.add_argument("--max_new_tokens_direct", type=int, default=64)
    parser.add_argument("--max_new_tokens_cot", type=int, default=1024)
    # ONE-PASS DESIGN: residual extraction happens here, not in a separate
    # GPU job. Default ON; opt out only for runs that will never be probed.
    parser.add_argument("--no_save_hidden_states", action="store_true",
                        help="Skip saving answer-prefix residuals during direct generation.")
    parser.add_argument("--hs_chunk", type=int, default=2,
                        help="Sub-batch size for the hidden-state forward (memory-bound on long prompts).")
    args = parser.parse_args()

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(
            f"Invalid shard config: shard_index={args.shard_index}, num_shards={args.num_shards}"
        )

    data_path, dataset_tag = resolve_data_path(args.dataset_name, args.set_id)
    model_short_name = args.model_name.split("/")[-1] + "_transformers"
    out_path = resolve_output_path(dataset_tag, model_short_name)
    if args.num_shards > 1:
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}.shard{args.shard_index}of{args.num_shards}{ext}"

    print(f"Resolved data_path: {data_path}")
    print(f"Resolved out_path : {out_path}")

    if os.path.exists(out_path):
        df = pd.read_csv(out_path, encoding="utf-8")
        if args.overwrite:
            tasks_to_run = list(args.tasks)
            print(f"Loaded existing results; --overwrite set, regenerating: {tasks_to_run} (other columns preserved)")
        else:
            tasks_to_run = [task for task in args.tasks if column_missing_or_incomplete(df, task)]
            needs_evaluation = any(
                output_column in df.columns
                and (
                    column_missing_or_incomplete(df, f"{output_column}_answer")
                    or column_missing_or_incomplete(df, f"{output_column}_correctness")
                )
                for output_column in args.tasks
            )
            if not tasks_to_run and not needs_evaluation:
                print(f"Result file already has requested outputs and overwrite is disabled: {out_path}")
                return
            print(f"Loaded existing results; missing generation columns: {tasks_to_run}")
    else:
        df = pd.read_csv(data_path, encoding="utf-8")
        if args.num_shards > 1:
            n = len(df)
            lo = (n * args.shard_index) // args.num_shards
            hi = (n * (args.shard_index + 1)) // args.num_shards
            df = df.iloc[lo:hi].reset_index(drop=True)
            print(f"Shard {args.shard_index}/{args.num_shards}: rows [{lo}:{hi}) -> {len(df)} rows")
        tasks_to_run = args.tasks

    print(f"Loaded dataframe with columns: {list(df.columns)}")
    model = None
    tokenizer = None
    if tasks_to_run:
        model, tokenizer = load_model_transformers(args.model_name)
        print(f"Loaded tokenizer pad_token_id={tokenizer.pad_token_id}, padding_side={tokenizer.padding_side}")

    print("=====================================================================")
    print("Starting transformers original-answer inference job...")
    print(f"Model  : {model_short_name}")
    print(f"Dataset: {dataset_tag}")
    print(f"Rows   : {len(df)}")
    print("=====================================================================")

    text_answers = args.dataset_name in TEXT_ANSWER_DATASETS
    task_cfg_key = "original_text" if text_answers else "original"
    evaluate = evaluate_final_answer_text if text_answers else evaluate_final_answer

    df = df.copy()
    if "original_direct" in tasks_to_run:
        capture = not args.no_save_hidden_states
        df["original_direct"], hidden = generate_original(
            df=df,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            instruction_key="direct",
            output_column="original_direct",
            max_new_tokens=args.max_new_tokens_direct,
            task_cfg_key=task_cfg_key,
            capture_hidden=capture,
            hs_chunk=args.hs_chunk,
        )
        if hidden is not None:
            import numpy as np
            hs_path = hidden_states_path(
                dataset_tag, model_short_name, args.num_shards, args.shard_index)
            np.savez_compressed(hs_path, hidden_states=hidden)
            print(f"Saved answer-prefix residuals {hidden.shape} to {hs_path}")
    if "original_cot" in tasks_to_run:
        df["original_cot"], _ = generate_original(
            df=df,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            instruction_key="cot",
            output_column="original_cot",
            max_new_tokens=args.max_new_tokens_cot,
            task_cfg_key=task_cfg_key,
        )
    print("Finished generation; starting answer extraction...")
    for output_column in ["original_direct", "original_cot"]:
        if output_column in df.columns:
            df[f"{output_column}_answer"] = df[output_column].apply(postprocess_generation)
    print("Finished answer extraction; starting correctness evaluation...")
    for output_column in ["original_direct", "original_cot"]:
        answer_column = f"{output_column}_answer"
        if answer_column in df.columns:
            df[f"{output_column}_correctness"] = df.apply(
                lambda r: evaluate(r[answer_column], r["answer"], verbose=False),
                axis=1,
            )
    print("Finished correctness evaluation; saving CSV...")
    df.to_csv(out_path, index=False)
    print(f"Saved transformers original-answer results to {out_path}")


CUDA_OOM_EXIT_CODE = 42


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


if __name__ == "__main__":
    import sys, traceback
    try:
        main()
    except BaseException as e:  # catch SystemExit too so OOM during shutdown is flagged
        if _is_cuda_oom(e):
            print(f"CUDA_OOM_DETECTED: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(CUDA_OOM_EXIT_CODE)
        raise
