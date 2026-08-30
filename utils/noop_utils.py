import os
import re
from functools import lru_cache
import torch
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import CACHE_DIR, DATA_DIR, HOME_DIR


def load_model(model_id, bfloat=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, cache_dir=CACHE_DIR, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if bfloat else torch.float16,
        cache_dir=CACHE_DIR,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    # Guard against silent CPU fallback: if any GPU on the host is already
    # contended, device_map="auto" can place layers on CPU and the run goes
    # ~1000x slower. Hard-fail instead.
    devs = {p.device.type for p in model.parameters()}
    if devs != {"cuda"}:
        import sys
        print(
            f"FATAL: model loaded on {devs} (expected cuda-only); "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}. "
            f"Exiting 42 for requeue.",
            file=sys.stderr, flush=True,
        )
        sys.exit(42)
    return model, tokenizer


DATASET_CHOICES = {
    "all":                    "noop_patching_data.csv",
    "sym_abs_wrong":          "noop_patching_data_sym_abs_wrong.csv",
    "sym_abs_correct":        "noop_patching_data_sym_abs_correct.csv",
    "direct_all":             "direct_patching_data.csv",
    "direct_sym_abs_wrong":   "direct_patching_data_sym_abs_wrong.csv",
    "direct_sym_abs_correct": "direct_patching_data_sym_abs_correct.csv",
    # gsm_symbolic original_direct_correct filtered, split by noop_correct × sym_abs
    "noop_correct_sym_abs_correct": "gsm_noop_hierarchy/direct/noop_correct_sym_abs_correct.csv",
    "noop_correct_sym_abs_wrong":   "gsm_noop_hierarchy/direct/noop_correct_sym_abs_wrong.csv",
    "noop_wrong_sym_abs_correct":   "gsm_noop_hierarchy/direct/noop_wrong_sym_abs_correct.csv",
    "noop_wrong_sym_abs_wrong":     "gsm_noop_hierarchy/direct/noop_wrong_sym_abs_wrong.csv",
    # gsm_symbolic original_cot_correct filtered, split by noop_cot_correct × sym_abs_cot
    "cot_noop_correct_sym_abs_correct": "gsm_noop_hierarchy/cot/noop_correct_sym_abs_correct.csv",
    "cot_noop_correct_sym_abs_wrong":   "gsm_noop_hierarchy/cot/noop_correct_sym_abs_wrong.csv",
    "cot_noop_wrong_sym_abs_correct":   "gsm_noop_hierarchy/cot/noop_wrong_sym_abs_correct.csv",
    "cot_noop_wrong_sym_abs_wrong":     "gsm_noop_hierarchy/cot/noop_wrong_sym_abs_wrong.csv",
}

_COT_SWAP_DIR = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap"
_COT_SWAP_NOOP_CLEAN_DIR = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_noop_clean"
_COT_SWAP_FILLER_VS_NOOP_CLEAN_DIR = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_filler_vs_noop_clean"
_COT_SWAP_FILLER_DF_CLEAN_VS_NOOP_CLEAN_FAIL_DIR = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_filler_df_clean_vs_noop_clean_fail"
_COT_SWAP_FILLER_DF_CORRECT_VS_NOOP_CLEAN_WRONG_DIR = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_filler_df_correct_vs_noop_clean_wrong"
_COT_SWAP_FILLER_DF_ALL_VS_NOOP_CLEAN_ALL_DIR       = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_filler_df_all_vs_noop_clean_all"
_COT_SWAP_FILLER_CLEAN_VS_NOOP_CLEAN_FAIL_DIR       = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_filler_clean_vs_noop_clean_fail"
_COT_SWAP_FILLER_CORRECT_VS_NOOP_CLEAN_WRONG_DIR    = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_filler_correct_vs_noop_clean_wrong"
_COT_SWAP_NOOP_CLEAN_WITHIN_TEMPLATE_DIR            = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_noop_clean_within_template"
_COT_SWAP_NOOP_CLEAN_WITHIN_TEMPLATE_CLEAN_VS_FAIL_DIR = Path(HOME_DIR) / "results/disentangled_evaluation/cot_swap_noop_clean_within_template_clean_vs_fail"
_COT_SWAP_MODEL_SUFFIX = "Llama-3.3-70B-Instruct"


def set_cot_swap_model_suffix(model_id_or_suffix: str) -> str:
    """Set the model suffix used to glob inference/cot_swap CSVs.

    CSV filenames carry the HF model basename (e.g. `Llama-3.3-70B-Instruct`,
    `gemma-2-9b-it`), so pass either a full model_id (`google/gemma-2-9b-it`)
    or the bare suffix. Call this once at startup from the run script so the
    loaders track `--model_id` instead of the hardcoded default.
    """
    global _COT_SWAP_MODEL_SUFFIX
    _COT_SWAP_MODEL_SUFFIX = model_id_or_suffix.split("/")[-1]
    return _COT_SWAP_MODEL_SUFFIX
_COT_SWAP_SOURCES = {
    # source name -> (directory, file_prefix)
    "noop":                 (_COT_SWAP_DIR,                       "cot_swap_original_set_"),
    "noop_clean":           (_COT_SWAP_NOOP_CLEAN_DIR,            "cot_swap_noop_clean_set_"),
    "filler_vs_noop_clean": (_COT_SWAP_FILLER_VS_NOOP_CLEAN_DIR,  "cot_swap_filler_vs_noop_clean_set_"),
    # Patching-only rebuild: clean filler_df (correct=T ∧ hash=T) joined with
    # failed noop_clean (correct=F ∧ hash=F). No swap inference run on this
    # CSV — `swap_correct` is absent. Only usable via
    # load_cot_swap_as_patching_pairs / patching downstream.
    "filler_df_clean_vs_noop_clean_fail": (
        _COT_SWAP_FILLER_DF_CLEAN_VS_NOOP_CLEAN_FAIL_DIR,
        "cot_swap_filler_df_clean_vs_noop_clean_fail_set_",
    ),
    # Same pair shape, looser selection: correct-side joined with wrong-side,
    # without the `####`-immediate requirement. Built by
    # create_cot_swap_patching_pair_data.py --filter correct_vs_wrong.
    "filler_df_correct_vs_noop_clean_wrong": (
        _COT_SWAP_FILLER_DF_CORRECT_VS_NOOP_CLEAN_WRONG_DIR,
        "cot_swap_filler_df_correct_vs_noop_clean_wrong_set_",
    ),
    # No filter on either side beyond the intrinsic `original_cot_correctness=F`
    # of cot_swap_noop_clean. Used for divergence DLA where filler-DF is only
    # a distractor-number-id source; selection of noop_clean_wrong rows is
    # NOT filtered on `swap_correct` (irrelevant to natural-CoT mechanism).
    "filler_df_all_vs_noop_clean_all": (
        _COT_SWAP_FILLER_DF_ALL_VS_NOOP_CLEAN_ALL_DIR,
        "cot_swap_filler_df_all_vs_noop_clean_all_set_",
    ),
    # Same pair logic as the filler_df contrasts but using gsm_filler (with
    # digits) as the clean-side source. Useful when contrasting against a
    # filler distractor that contains numeric tokens, which gets the model to
    # re-reason more often than digit-free filler_df.
    "filler_clean_vs_noop_clean_fail": (
        _COT_SWAP_FILLER_CLEAN_VS_NOOP_CLEAN_FAIL_DIR,
        "cot_swap_filler_clean_vs_noop_clean_fail_set_",
    ),
    "filler_correct_vs_noop_clean_wrong": (
        _COT_SWAP_FILLER_CORRECT_VS_NOOP_CLEAN_WRONG_DIR,
        "cot_swap_filler_correct_vs_noop_clean_wrong_set_",
    ),
    # Self-paired noop_clean_wrong: same CSV as `noop_clean` but emitted with
    # source == target so head_scaling at cot_start runs the intervention on
    # every noop_clean_wrong row without joining against any filler-side
    # trajectory. Used to test cell sets (selected on filler_df DLA) on the
    # full noop_clean_wrong population.
    "noop_clean_wrong_self": (
        _COT_SWAP_NOOP_CLEAN_DIR,
        "cot_swap_noop_clean_set_",
    ),
    # Same source CSV as `noop_clean` but emitted with source_question =
    # sym_question (the un-clause symbolic problem) and target_question =
    # noop_question. Lets divergence DLA's `extract_nums(tgt_q) - extract_nums
    # (src_q)` distractor-id work on every noop_clean_wrong row without
    # depending on a filler-DF partner (filler-DF is digit-free so sym-side
    # set-difference is mathematically equivalent on the overlap; 40 extra
    # rows recovered from the filler-missing partition).
    "noop_clean_wrong_sym_diff": (
        _COT_SWAP_NOOP_CLEAN_DIR,
        "cot_swap_noop_clean_set_",
    ),
    # P1 (extra-step) vs padded-Symbolic (length-matched, no extra step). No
    # underlying cot_swap CSV — `load_cot_swap_as_patching_pairs` builds the
    # patching-pair frame directly by inner-joining the two sides' inference
    # results on (original_id, instance). The tuple values are unused; the
    # entry is kept here so this source name is recognized by the dispatch.
    "p1_correct_vs_padded": (
        None,
        None,
    ),
    # Plain GSM-Symbolic, CoT-correct only, self-paired. Head-scaling control
    # to test whether ablating sym_diff cells damages clean Symbolic reasoning
    # (vs the noop / filler-DF results already in hand). Built directly from
    # transformers_direct/gsm_symbolic/ inference CSVs — no cot_swap CSV.
    "symbolic_correct_self": (
        None,
        None,
    ),
    # GSM-Filler (gsm_filler_df) CoT-correct only, self-paired. Damage-side
    # control at the full filler population (~4700 pairs / 100 templates),
    # without the noop-wrong join that capped the prior runs at 502 pairs.
    "filler_correct_self": (
        None,
        None,
    ),
}
_NOOP_REVIEW_PATH  = Path(HOME_DIR) / "data" / "gsm_symbolic" / "test_gsm_noop_review.csv"
_NOOP_SOURCE_DIR   = Path(HOME_DIR) / "data" / "gsm_symbolic" / "split_datasets_noop"


def load_cot_swap_data(drop_swap_correct_filter: bool = False, source: str = "noop"):
    """Load noop cot-swap rows.

    source:
      "noop" (default): the legacy CSV at results/disentangled_evaluation/cot_swap/
        — built from vLLM on the original gsm_noop wording. Has `swap_correct`.
      "noop_clean": the rebuild at results/disentangled_evaluation/cot_swap_noop_clean/
        — built from transformers_direct on gsm_noop_clean. `swap_correct` is
        NaN until disentangled_evaluation/cot_swap_experiment.py is run for the
        (symbolic, noop_clean) crossing; once populated, this source supports
        the `~swap_correct` filter just like "noop".
      "filler_vs_noop_clean": same story but for the (filler, noop_clean)
        crossing.

    drop_swap_correct_filter:
      False (default): keep only rows where [noop_q + sym_CoT] failed
        (`~swap_correct`) AND `sym_original_cot_correctness`. Original headline
        population. Requires `swap_correct` to be populated for the chosen
        source (run cot_swap_experiment.py first).
      True: keep all rows where the natural sym run was correct, regardless of
        swap outcome.
    """
    if source not in _COT_SWAP_SOURCES:
        raise ValueError(
            f"Unknown cot_swap source `{source}`. Choices: {list(_COT_SWAP_SOURCES)}"
        )
    cot_dir, prefix = _COT_SWAP_SOURCES[source]
    frames = []
    for f in sorted(cot_dir.glob(f"{prefix}*_{_COT_SWAP_MODEL_SUFFIX}.csv")):
        df = pd.read_csv(f)
        df["split"] = int(str(f).split("_set_")[1].split("_")[0])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No cot_swap CSVs in {cot_dir} matching `{prefix}*_{_COT_SWAP_MODEL_SUFFIX}.csv`"
        )
    df = pd.concat(frames, ignore_index=True)
    sym_ok = df["sym_original_cot_correctness"]
    if drop_swap_correct_filter:
        return df[sym_ok].copy()
    if df["swap_correct"].isna().any():
        raise ValueError(
            f"cot_swap source={source} has NaN `swap_correct` entries; cannot "
            "filter on `~swap_correct`. Run disentangled_evaluation/"
            "cot_swap_experiment.py for this crossing to populate the column, "
            "or pass drop_swap_correct_filter=True (--include_recovered)."
        )
    return df[~df["swap_correct"].astype(bool) & sym_ok].copy()


_COT_SWAP_SOURCE_LABELS = {
    "noop":                 ("symbolic",        "noop"),
    "noop_clean":           ("symbolic",        "noop_clean"),
    "filler_vs_noop_clean": ("filler",          "noop_clean"),
    "filler_df_clean_vs_noop_clean_fail":    ("filler_df_clean",   "noop_clean_fail"),
    "filler_df_correct_vs_noop_clean_wrong": ("filler_df_correct", "noop_clean_wrong"),
    "filler_df_all_vs_noop_clean_all":       ("filler_df_all",     "noop_clean_all"),
    "filler_clean_vs_noop_clean_fail":       ("filler_clean",      "noop_clean_fail"),
    "filler_correct_vs_noop_clean_wrong":    ("filler_correct",    "noop_clean_wrong"),
    "noop_clean_wrong_self":                 ("noop_clean_wrong",  "noop_clean_wrong"),
    "noop_clean_wrong_sym_diff":             ("symbolic",          "noop_clean_wrong"),
    "p1_correct_vs_padded":                  ("padded_symbolic",   "p1_correct"),
    "symbolic_correct_self":                 ("symbolic_correct",  "symbolic_correct"),
    "filler_correct_self":                   ("filler_correct",    "filler_correct"),
}


_NOOP_CLEAN_WITHIN_TEMPLATE_DIRS = {
    "noop_clean_within_template":               _COT_SWAP_NOOP_CLEAN_WITHIN_TEMPLATE_DIR,
    "noop_clean_within_template_clean_vs_fail": _COT_SWAP_NOOP_CLEAN_WITHIN_TEMPLATE_CLEAN_VS_FAIL_DIR,
}


def _load_noop_clean_within_template_pairs(source):
    """Load the noop_clean within-template pairs CSV (loose or strict filter).

    Both source names share the patching-pair schema (source_question /
    target_question / source_cot_prefix / target_cot_prefix / source_answer /
    target_answer / ...). We just concatenate the per-set files and pass them
    through. The only difference between the two sources is the filter that
    was applied at pair-build time (correct_vs_wrong vs clean_vs_fail).
    """
    cot_dir = _NOOP_CLEAN_WITHIN_TEMPLATE_DIRS[source]
    prefix = f"cot_swap_{source}"
    files = sorted(cot_dir.glob(f"{prefix}_set_*_{_COT_SWAP_MODEL_SUFFIX}.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSVs in {cot_dir} matching `{prefix}_set_*_{_COT_SWAP_MODEL_SUFFIX}.csv`. "
            "Generate the corresponding noop-clean within-template pair CSVs "
            "with the matching filter first."
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["source_question", "target_question", "source_cot_prefix", "target_cot_prefix"])
    return df.sort_values(["original_id", "instance"]).reset_index(drop=True)


# For each cot_swap CSV, where do each side's OWN CoTs live (inference results
# on the respective standalone dataset)? Used by `load_cot_swap_as_patching_pairs`
# to populate `source_own_cot_prefix` and `target_own_cot_prefix`, which the
# baseline computation uses so gold logprobs reflect each side's natural
# (model's own CoT) behavior, not a shared-donor-CoT artefact.
#
# Each value is (src_inference_subdir, tgt_inference_subdir) — both relative
# to `results/disentangled_evaluation/transformers_direct/`.
_OWN_COT_SOURCES = {
    "filler_df_correct_vs_noop_clean_wrong": ("gsm_filler_df", "gsm_noop_clean"),
    "filler_df_all_vs_noop_clean_all":       ("gsm_filler_df", "gsm_noop_clean"),
    "noop_clean_wrong_sym_diff":             ("gsm_symbolic",  "gsm_noop_clean"),
    "filler_correct_vs_noop_clean_wrong":    ("gsm_filler",    "gsm_noop_clean"),
    "filler_df_clean_vs_noop_clean_fail":    ("gsm_filler_df", "gsm_noop_clean"),
    "filler_clean_vs_noop_clean_fail":       ("gsm_filler",    "gsm_noop_clean"),
    "noop":                                  ("gsm_symbolic",  "gsm_noop"),
    "noop_clean":                            ("gsm_symbolic",  "gsm_noop_clean"),
    "filler_vs_noop_clean":                  ("gsm_filler",    "gsm_noop_clean"),
    "p1_correct_vs_padded":                  ("gsm_padded_symbolic_p1_aligned", "gsm_p1"),
    "symbolic_correct_self":                 ("gsm_symbolic", "gsm_symbolic"),
    "filler_correct_self":                   ("gsm_filler_df", "gsm_filler_df"),
}


def _load_own_cot_by_key(subdir, model_suffix=None):
    """Load per-(oid, instance) original_cot from inference CSVs. Returns a
    dict {(oid, instance): original_cot_string}.

    Looks under `results/disentangled_evaluation/transformers_direct/{subdir}/`
    for `*_results_{model_suffix}_transformers.csv` files.
    """
    if model_suffix is None:
        model_suffix = _COT_SWAP_MODEL_SUFFIX
    base = Path(HOME_DIR) / "results/disentangled_evaluation/transformers_direct" / subdir
    files = sorted(base.glob(f"*_results_{model_suffix}_transformers.csv"))
    if not files:
        return {}
    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["original_id", "instance", "original_cot"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return dict(
        zip(
            zip(df.original_id.astype(int), df.instance.astype(int)),
            df.original_cot.astype(str),
        )
    )


def load_cot_swap_as_patching_pairs(source: str = "filler_vs_noop_clean",
                                    cot_source: str | None = None):
    """Reshape a cot_swap CSV into the activation-patching schema.

    Returns columns expected by run_cot_swap_activation_patching.patch_one:
      original_id, instance, source_question, target_question,
      source_answer, target_answer, len_delta_after,
      source_label, target_label, source_cot_prefix
      [+ target_cot_prefix for within-template contrasts].

    Source-side question/CoT come from the `sym_*` columns of the cot_swap
    CSV (which for `filler_vs_noop_clean` actually hold the *filler* side,
    by schema convention). Target side is the noop question. Both source
    and target answers equal `sym_answer` (the gold answer; same problem,
    only the inserted clause differs).

    `cot_source` (default `None` → use `source`'s CoT): if specified to a
    DIFFERENT cot_swap source, the source_cot_prefix is replaced with that
    source's `sym_reasoning` joined by (original_id, instance). Use this
    to e.g. keep filler-DF/noop questions (length-controlled) but swap in
    symbolic's clean CoT as the donor reasoning.
    """
    if source in _NOOP_CLEAN_WITHIN_TEMPLATE_DIRS:
        if cot_source is not None and cot_source != source:
            raise ValueError(
                "cot_source override is not supported for within-template "
                "contrasts (target_cot_prefix is already per-pair)."
            )
        return _load_noop_clean_within_template_pairs(source)

    if source == "symbolic_correct_self":
        # Self-paired Symbolic-CoT-correct: source_question == target_question
        # == sym_question. Built directly from transformers_direct/gsm_symbolic/
        # inference CSVs (gsm_symbolic_set_*_*_transformers.csv only — exclude
        # gsm_symbolic_padded_*, gsm_symbolic_to_p1_* siblings sharing the dir).
        # Filter: original_cot_correctness == True. Only meaningful at
        # patch_position == "cot_start" (no donor CoT, model generates own CoT
        # post-scaling). Mirrors noop_clean_wrong_self for the clean Symbolic
        # baseline.
        if cot_source is not None and cot_source != source:
            raise ValueError(
                "cot_source override is not supported for symbolic_correct_self "
                "(there is no donor CoT in this contrast)."
            )
        from run_cot_swap_logit_lens import cot_prefix_before_hash
        src_label, tgt_label = _COT_SWAP_SOURCE_LABELS[source]
        td_root = Path(HOME_DIR) / "results/disentangled_evaluation/transformers_direct"
        base = td_root / "gsm_symbolic"
        files = sorted(base.glob(f"gsm_symbolic_set_*_{_COT_SWAP_MODEL_SUFFIX}_transformers.csv"))
        if not files:
            raise FileNotFoundError(
                f"No gsm_symbolic_set_* CSVs in {base} for {_COT_SWAP_MODEL_SUFFIX}"
            )
        sym = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        sym = sym[sym["original_cot_correctness"] == True].copy()

        def _cot_prefix(s):
            return (cot_prefix_before_hash(str(s)) or "").rstrip() + "\n\n"

        out = pd.DataFrame({
            "original_id":      sym["original_id"].astype(int),
            "instance":         sym["instance"].astype(int),
            "source_question":  sym["question"].astype(str),
            "target_question":  sym["question"].astype(str),
            "source_answer":    sym["answer"].astype(str),
            "target_answer":    sym["answer"].astype(str),
            "len_delta_after":  0,
            "source_label":     src_label,
            "target_label":     tgt_label,
            "source_cot_prefix":     "",
            "source_own_cot_prefix": [_cot_prefix(c) for c in sym["original_cot"]],
            "target_own_cot_prefix": [_cot_prefix(c) for c in sym["original_cot"]],
        })
        return out.dropna(
            subset=["target_question"]
        ).sort_values(["original_id", "instance"]).reset_index(drop=True)

    if source == "filler_correct_self":
        # Self-paired GSM-Filler-CoT-correct: source_question == target_question
        # == filler_df_question. Built directly from transformers_direct/
        # gsm_filler_df/ inference CSVs. Filter: original_cot_correctness == True.
        # Only meaningful at patch_position == "cot_start" (no donor CoT, model
        # generates own CoT post-scaling). Mirrors symbolic_correct_self but
        # for the digit-free filler population (~4700 pairs / 100 templates),
        # replacing the smaller filler_df_correct_vs_noop_clean_wrong join used
        # in earlier damage runs.
        if cot_source is not None and cot_source != source:
            raise ValueError(
                "cot_source override is not supported for filler_correct_self "
                "(there is no donor CoT in this contrast)."
            )
        from run_cot_swap_logit_lens import cot_prefix_before_hash
        src_label, tgt_label = _COT_SWAP_SOURCE_LABELS[source]
        td_root = Path(HOME_DIR) / "results/disentangled_evaluation/transformers_direct"
        base = td_root / "gsm_filler_df"
        files = sorted(base.glob(f"gsm_filler_df_set_*_{_COT_SWAP_MODEL_SUFFIX}_transformers.csv"))
        if not files:
            raise FileNotFoundError(
                f"No gsm_filler_df_set_* CSVs in {base} for {_COT_SWAP_MODEL_SUFFIX}"
            )
        fil = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        fil = fil[fil["original_cot_correctness"] == True].copy()

        def _cot_prefix(s):
            return (cot_prefix_before_hash(str(s)) or "").rstrip() + "\n\n"

        out = pd.DataFrame({
            "original_id":      fil["original_id"].astype(int),
            "instance":         fil["instance"].astype(int),
            "source_question":  fil["question"].astype(str),
            "target_question":  fil["question"].astype(str),
            "source_answer":    fil["answer"].astype(str),
            "target_answer":    fil["answer"].astype(str),
            "len_delta_after":  0,
            "source_label":     src_label,
            "target_label":     tgt_label,
            "source_cot_prefix":     "",
            "source_own_cot_prefix": [_cot_prefix(c) for c in fil["original_cot"]],
            "target_own_cot_prefix": [_cot_prefix(c) for c in fil["original_cot"]],
        })
        return out.dropna(
            subset=["target_question"]
        ).sort_values(["original_id", "instance"]).reset_index(drop=True)

    if source == "p1_correct_vs_padded":
        # P1 (target, extra-step) vs padded-Symbolic (source, length-matched).
        # Built directly from each side's transformers_direct inference CSVs
        # — no cot_swap intermediate. Filter: P1 correct (target side; needed
        # so the natural CoT engages with the extra step); no filter on the
        # padded side (mirrors the divergence-DLA spec where the source-side
        # is only used for set-diff number identification).
        from run_cot_swap_logit_lens import cot_prefix_before_hash
        src_label, tgt_label = _COT_SWAP_SOURCE_LABELS[source]
        src_subdir, tgt_subdir = _OWN_COT_SOURCES[source]
        td_root = Path(HOME_DIR) / "results/disentangled_evaluation/transformers_direct"

        def _load_inference(subdir):
            base = td_root / subdir
            files = sorted(base.glob(f"*_results_{_COT_SWAP_MODEL_SUFFIX}_transformers.csv"))
            if not files:
                raise FileNotFoundError(
                    f"No inference CSVs in {base} matching "
                    f"`*_results_{_COT_SWAP_MODEL_SUFFIX}_transformers.csv`"
                )
            return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

        p1 = _load_inference(tgt_subdir)
        pad = _load_inference(src_subdir)
        # P1✓ rows only; padded side intentionally unfiltered.
        p1 = p1[p1["original_cot_correctness"] == True].copy()
        merged = p1[["original_id", "instance", "question", "answer", "original_cot"]].rename(
            columns={"question": "p1_question",
                     "original_cot": "p1_cot"}
        ).merge(
            pad[["original_id", "instance", "question", "original_cot"]].rename(
                columns={"question": "padded_question",
                         "original_cot": "padded_cot"}
            ),
            on=["original_id", "instance"], how="inner",
        )

        def _cot_prefix(s):
            return (cot_prefix_before_hash(str(s)) or "").rstrip() + "\n\n"

        out = pd.DataFrame({
            "original_id":      merged["original_id"].astype(int),
            "instance":         merged["instance"].astype(int),
            "source_question":  merged["padded_question"].astype(str),
            "target_question":  merged["p1_question"].astype(str),
            "source_answer":    merged["answer"].astype(str),
            "target_answer":    merged["answer"].astype(str),
            "len_delta_after":  0,
            "source_label":     src_label,
            "target_label":     tgt_label,
            "source_cot_prefix":     [_cot_prefix(c) for c in merged["padded_cot"]],
            "source_own_cot_prefix": [_cot_prefix(c) for c in merged["padded_cot"]],
            "target_own_cot_prefix": [_cot_prefix(c) for c in merged["p1_cot"]],
        })
        return out.dropna(
            subset=["source_question", "target_question", "target_own_cot_prefix"]
        ).sort_values(["original_id", "instance"]).reset_index(drop=True)

    if source == "noop_clean_wrong_sym_diff":
        # source_question = sym_question (the un-clause symbolic problem),
        # target_question = noop_question. Designed for divergence DLA whose
        # `extract_nums(target_question) - extract_nums(source_question)` then
        # yields the noop-clause-only numbers — independent of any filler-DF
        # partner. Covers all 936 noop_clean_wrong rows × 44 templates.
        # cot_source override is accepted but redundant: this source already
        # reads sym_reasoning from the noop_clean CSV (same data the
        # cot_source="noop_clean" override would have fetched). Only reject
        # genuinely incompatible cot_source values.
        if cot_source is not None and cot_source not in (source, "noop_clean"):
            raise ValueError(
                f"cot_source={cot_source!r} not supported for noop_clean_wrong_sym_diff; "
                f"valid options are None or 'noop_clean' (both equivalent here)."
            )
        src_label, tgt_label = _COT_SWAP_SOURCE_LABELS[source]
        df = load_cot_swap_data(drop_swap_correct_filter=True, source="noop_clean")
        out = pd.DataFrame({
            "original_id":      df["id"].astype(int),
            "instance":         df["instance"].astype(int),
            "source_question":  df["sym_question"].astype(str),
            "target_question":  df["noop_question"].astype(str),
            "source_answer":    df["sym_answer"].astype(str),
            "target_answer":    df["sym_answer"].astype(str),
            "len_delta_after":  0,
            "source_label":     src_label,
            "target_label":     tgt_label,
            "source_cot_prefix": df["sym_reasoning"].astype(str).str.rstrip() + "\n\n",
        })
        out = out.dropna(
            subset=["source_question", "target_question"]
        ).sort_values(["original_id", "instance"]).reset_index(drop=True)
        # Populate target_own_cot_prefix from transformers_direct/gsm_noop_clean
        # (the model's natural CoT on the noop side) — this is the trace
        # divergence DLA reads to locate the wrong-plan engagement clause.
        if source in _OWN_COT_SOURCES:
            from run_cot_swap_logit_lens import cot_prefix_before_hash
            src_subdir, tgt_subdir = _OWN_COT_SOURCES[source]
            src_own = _load_own_cot_by_key(src_subdir)
            tgt_own = _load_own_cot_by_key(tgt_subdir)
            keys = list(zip(out.original_id.astype(int), out.instance.astype(int)))
            out["source_own_cot_prefix"] = [
                (cot_prefix_before_hash(src_own.get(k, "")) or "").rstrip() + "\n\n"
                for k in keys
            ]
            out["target_own_cot_prefix"] = [
                (cot_prefix_before_hash(tgt_own.get(k, "")) or "").rstrip() + "\n\n"
                for k in keys
            ]
        return out

    if source == "noop_clean_wrong_self":
        # Self-paired schema: source == target == noop_clean question. No
        # filler-side trajectory, no donor CoT. Only meaningful at
        # patch_position="cot_start", where source-side log P(####) is
        # unused and the model generates its own CoT post-scaling.
        if cot_source is not None and cot_source != source:
            raise ValueError(
                "cot_source override is not supported for noop_clean_wrong_self "
                "(there is no donor CoT in this contrast)."
            )
        src_label, tgt_label = _COT_SWAP_SOURCE_LABELS[source]
        df = load_cot_swap_data(drop_swap_correct_filter=True, source="noop_clean")
        out = pd.DataFrame({
            "original_id":      df["id"].astype(int),
            "instance":         df["instance"].astype(int),
            "source_question":  df["noop_question"].astype(str),
            "target_question":  df["noop_question"].astype(str),
            "source_answer":    df["sym_answer"].astype(str),
            "target_answer":    df["sym_answer"].astype(str),
            "len_delta_after":  0,
            "source_label":     src_label,
            "target_label":     tgt_label,
            "source_cot_prefix": "",
        })
        return out.dropna(
            subset=["target_question"]
        ).sort_values(["original_id", "instance"]).reset_index(drop=True)

    src_label, tgt_label = _COT_SWAP_SOURCE_LABELS[source]
    df = load_cot_swap_data(drop_swap_correct_filter=True, source=source)
    out = pd.DataFrame({
        "original_id":      df["id"].astype(int),
        "instance":         df["instance"].astype(int),
        "source_question":  df["sym_question"].astype(str),
        "target_question":  df["noop_question"].astype(str),
        "source_answer":    df["sym_answer"].astype(str),
        "target_answer":    df["sym_answer"].astype(str),
        "len_delta_after":  0,
        "source_label":     src_label,
        "target_label":     tgt_label,
        "source_cot_prefix": df["sym_reasoning"].astype(str),
    })

    if cot_source is not None and cot_source != source:
        # Replace source_cot_prefix with the CoT from a different contrast,
        # joined by (original_id, instance). Rows without a matching CoT
        # entry are dropped — the result is a (questions from `source`) ×
        # (CoT from `cot_source`) hybrid.
        cot_df = load_cot_swap_data(drop_swap_correct_filter=True, source=cot_source)
        cot_join = cot_df[["id", "instance", "sym_reasoning"]].rename(
            columns={"id": "original_id", "sym_reasoning": "_cot_from_alt"}
        ).astype({"original_id": int, "instance": int})
        before = len(out)
        out = out.drop(columns=["source_cot_prefix"]).merge(
            cot_join, on=["original_id", "instance"], how="inner"
        ).rename(columns={"_cot_from_alt": "source_cot_prefix"})
        dropped = before - len(out)
        if dropped:
            print(
                f"load_cot_swap_as_patching_pairs: dropped {dropped}/{before} "
                f"rows lacking a matching `{cot_source}` CoT entry."
            )

    out = out.dropna(subset=["source_question", "target_question", "source_cot_prefix"])
    # EOL normalization: the cot_swap CSV's `sym_reasoning` column has trailing
    # whitespace stripped — without adding it back, the last token of the
    # cot_end prompt sits at a period like "$948." and log P(####) is far from
    # the "model is primed to emit ####" position. Re-append the canonical
    # "\n\n" separator that immediately precedes `####` in the original CoT,
    # matching what `cot_prefix_before_hash` produces for the p1 pair loader.
    # Downstream consumers (head_patching, head_scaling, activation patching)
    # all read source_cot_prefix verbatim, so this fixes the readout for ALL
    # of them at once.
    out["source_cot_prefix"] = out["source_cot_prefix"].str.rstrip() + "\n\n"

    # Each side's OWN CoT (model's natural generation on that side's question).
    # Used by `compute_cot_swap_baselines` so gold-logprob baselines reflect
    # the natural cliff (filler→correct, noop→wrong) rather than the shared-
    # donor-CoT artefact. Pulled from per-side inference CSVs; rows missing
    # an own-CoT entry on either side get NaN, which downstream code can
    # detect.
    if source in _OWN_COT_SOURCES:
        from run_cot_swap_logit_lens import cot_prefix_before_hash
        src_subdir, tgt_subdir = _OWN_COT_SOURCES[source]
        src_own = _load_own_cot_by_key(src_subdir)
        tgt_own = _load_own_cot_by_key(tgt_subdir)
        keys = list(zip(out.original_id.astype(int), out.instance.astype(int)))
        out["source_own_cot_prefix"] = [
            cot_prefix_before_hash(src_own.get(k, "")) or "" for k in keys
        ]
        out["target_own_cot_prefix"] = [
            cot_prefix_before_hash(tgt_own.get(k, "")) or "" for k in keys
        ]
        # Match the EOL convention used for source_cot_prefix above.
        out["source_own_cot_prefix"] = (
            out["source_own_cot_prefix"].str.rstrip() + "\n\n"
        )
        out["target_own_cot_prefix"] = (
            out["target_own_cot_prefix"].str.rstrip() + "\n\n"
        )
    return out.sort_values(["original_id", "instance"]).reset_index(drop=True)


# cot_boundary baselines: noop natural generation under transformers
# re-generation. Used by `load_flipped_pair_ids` and `filter_flipped_pairs`
# to drop pairs where noop's natural completion produces gold despite being
# labeled noop_clean_wrong (the original filter used a decoding pipeline that
# differs slightly from our analysis re-generation).
_FLIPPED_BASELINE_PATH = (
    Path(HOME_DIR) / "results" / "noop_patching" / "llama-3.3-70b-instruct"
    / "filler_df_vs_noop_clean_tfm" / "cot_boundary" / "question_span"
    / "baselines_correct_to_wrong.jsonl"
)


def load_flipped_pair_ids():
    """Return set of (original_id, instance) keys for pairs where noop's
    natural re-generation produces the gold answer. Affects ~2.7% of
    cot_boundary baseline pairs."""
    import json as _json
    flipped = set()
    if not _FLIPPED_BASELINE_PATH.exists():
        print(f"warning: flipped-baseline file missing at {_FLIPPED_BASELINE_PATH}")
        return flipped
    with open(_FLIPPED_BASELINE_PATH) as _f:
        for line in _f:
            r = _json.loads(line)
            if r.get("unpatched_target_correct"):
                flipped.add((int(r["original_id"]), int(r["instance_w"])))
    return flipped


def filter_flipped_pairs(df):
    """Drop rows whose (original_id, instance) is in load_flipped_pair_ids().
    Returns a new DataFrame; the input is not mutated."""
    flipped = load_flipped_pair_ids()
    if not flipped:
        return df.reset_index(drop=True)
    keep = df.apply(lambda r: (int(r["original_id"]), int(r["instance"])) not in flipped, axis=1)
    return df[keep].reset_index(drop=True)


def to_serializable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    elif hasattr(x, "item"):
        return x.item()
    return x


_PLACEHOLDER_RE = re.compile(r"\b[a-z]\b")
_BINDING_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,]+)")
_NUMERIC_VALUE_RE = re.compile(r"\d")
_NUMERIC_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "half", "halves", "quarter",
    "quarters", "third", "thirds", "fourth", "fourths", "fifth", "fifths",
    "sixth", "sixths", "seventh", "sevenths", "eighth", "eighths", "ninth",
    "ninths", "tenth", "tenths", "eleventh", "elevenths", "twelfth",
    "twelfths", "dozen",
}
_NUMERIC_CONNECTORS = {"and", "a", "an", "of"}
_NUMERIC_PHRASE_RE = re.compile(
    r"""
    (?:
        \b\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\b
        |
        \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|
           thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|
           thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|
           billion|half|halves|quarter|quarters|third|thirds|fourth|fourths|fifth|
           fifths|sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|
           tenths|eleventh|elevenths|twelfth|twelfths|dozen)(?:[-\s]+
           (?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|
           thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|
           thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|
           billion|half|halves|quarter|quarters|third|thirds|fourth|fourths|fifth|
           fifths|sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|
           tenths|eleventh|elevenths|twelfth|twelfths|dozen|and|a|an|of))*
        \b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_DIGITISH_RE = re.compile(r"\d")


def parse_symbol_bindings(symbol_binding):
    """Parse `x = 3, y = 7` into `{'x': '3', 'y': '7'}`."""
    if not isinstance(symbol_binding, str):
        return {}
    return {
        key.strip(): value.strip()
        for key, value in _BINDING_RE.findall(symbol_binding)
    }


@lru_cache(maxsize=1)
def load_noop_clause_map():
    """Load the reviewed noop clause keyed by original_id."""
    if not _NOOP_REVIEW_PATH.exists():
        return {}

    review = pd.read_csv(_NOOP_REVIEW_PATH).dropna(subset=["noop_clause"])
    return {
        int(oid): clause.strip()
        for oid, clause in zip(review["original_id"], review["noop_clause"])
    }


def extract_expression_variables(expr):
    """Return symbolic variable names used in an abstraction expression."""
    if not isinstance(expr, str):
        return set()

    expr = re.sub(r"\bWhat is the value of\b", "", expr)
    candidates = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", expr))
    return {
        token for token in candidates
        if token not in {"What", "is", "the", "value", "of"}
    }


def extract_question_placeholders(symbolic_question):
    """Return placeholder variables from a symbolic question template."""
    if not isinstance(symbolic_question, str):
        return []

    placeholders = []
    for match in _PLACEHOLDER_RE.finditer(symbolic_question):
        token = match.group(0)
        prev_char = symbolic_question[match.start() - 1] if match.start() > 0 else " "
        next_char = (
            symbolic_question[match.end()]
            if match.end() < len(symbolic_question) else " "
        )
        if prev_char.isalnum() or next_char.isalnum():
            continue
        placeholders.append(token)
    return placeholders


def _looks_numeric_value(value):
    if not isinstance(value, str):
        return False

    value = value.strip()
    if not value:
        return False
    if _NUMERIC_VALUE_RE.search(value):
        return True

    cleaned = re.sub(r"[-]", " ", value.lower())
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens:
        return False
    return all(tok in _NUMERIC_WORDS or tok in _NUMERIC_CONNECTORS for tok in tokens)


def normalize_numeric_value(value):
    """Return a clean numeric-like value string, or None if the value is not usable.

    Accepts digit-based values (including decimals, fractions, percents) and
    fully textual numeric phrases like ``thirty five``. Rejects mixed captures
    such as ``one 42`` and unit strings like ``meters``.
    """
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None
    if not _looks_numeric_value(value):
        return None

    has_alpha = bool(re.search(r"[A-Za-z]", value))
    has_digit = bool(_DIGITISH_RE.search(value))
    if has_alpha and has_digit:
        return None

    return value


def extract_numeric_phrases(text):
    """Extract numeric-looking phrases from free text."""
    if not isinstance(text, str):
        return []

    digit_phrases = []
    word_phrases = []
    for match in _NUMERIC_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip(" ,.")
        if _looks_numeric_value(phrase):
            if _DIGITISH_RE.search(phrase):
                digit_phrases.append(phrase)
            else:
                word_phrases.append(phrase)
    return digit_phrases if digit_phrases else word_phrases


def extract_noop_clause(row):
    """Return the reviewed noop clause for a row's original_id, if available."""
    original_id = row.get("original_id")
    if original_id is None:
        return None
    try:
        original_id = int(original_id)
    except (TypeError, ValueError):
        return None
    return load_noop_clause_map().get(original_id)


def extract_noop_clause_values(row):
    """
    Return numeric-looking values that appear in the canonical reviewed noop
    clause for this original problem.
    """
    clause = extract_noop_clause(row)
    if clause is None:
        return []
    return extract_numeric_phrases(clause)


def extract_primary_noop_distractor_value(row):
    """
    Return a single distractor value from the canonical noop clause.

    If the clause contains multiple explicit numeric phrases, take the first one.
    """
    values = extract_noop_clause_values(row)
    return values[0] if values else None


def extract_noop_variable_values(question, symbolic_question):
    """
    Align a concrete noop question against its symbolic template and recover the
    realized string value for each placeholder variable.
    """
    if not isinstance(question, str) or not isinstance(symbolic_question, str):
        return {}

    placeholders = extract_question_placeholders(symbolic_question)
    if not placeholders:
        return {}

    pattern_parts = []
    last_idx = 0
    seen = {}

    for match in _PLACEHOLDER_RE.finditer(symbolic_question):
        token = match.group(0)
        if token not in placeholders:
            continue

        pattern_parts.append(re.escape(symbolic_question[last_idx:match.start()]))
        if token in seen:
            pattern_parts.append(fr"(?P={token})")
        else:
            pattern_parts.append(fr"(?P<{token}>.+?)")
            seen[token] = True
        last_idx = match.end()

    pattern_parts.append(re.escape(symbolic_question[last_idx:]))
    pattern = "".join(pattern_parts)
    matched = re.fullmatch(pattern, question, flags=re.IGNORECASE)
    if matched is None:
        return {}
    values = {}
    for key, value in matched.groupdict().items():
        value = normalize_numeric_value(value)
        if value is None:
            continue
        if value == key:
            continue
        values[key] = value
    return values


def extract_noop_distractor_variables(row):
    """
    Recover distractor placeholders and their concrete values for a noop row.

    A distractor variable is a placeholder that appears in the symbolic question
    but not in the actual computation expression.
    """
    bindings = parse_symbol_bindings(row.get("symbol_binding"))
    expr_vars = (
        extract_expression_variables(row.get("symbolic_abstraction_answer")) |
        extract_expression_variables(row.get("numerical_abstraction_answer")) |
        set(bindings)
    )
    values = extract_noop_variable_values(
        row.get("question"), row.get("symbolic_question")
    )
    distractor_vars = sorted(var for var in values if var not in expr_vars)
    distractor_values = {var: values[var] for var in distractor_vars}
    return distractor_vars, distractor_values


def extract_noop_relevant_variables(row):
    """
    Recover computation-relevant placeholders and their concrete values for a
    noop row.
    """
    bindings = parse_symbol_bindings(row.get("symbol_binding"))
    expr_vars = (
        extract_expression_variables(row.get("symbolic_abstraction_answer")) |
        extract_expression_variables(row.get("numerical_abstraction_answer")) |
        set(bindings)
    )
    values = extract_noop_variable_values(
        row.get("question"), row.get("symbolic_question")
    )
    relevant_vars = sorted(expr_vars & (set(values) | set(bindings)))
    relevant_values = {}
    for var in relevant_vars:
        raw_value = values.get(var, bindings.get(var))
        value = normalize_numeric_value(raw_value)
        if value is None:
            continue
        relevant_values[var] = value
    relevant_vars = [var for var in relevant_vars if var in relevant_values]
    return relevant_vars, relevant_values


def annotate_noop_distractors(df):
    """Add distractor variable/value columns to a noop dataframe copy."""
    annotated = df.copy()
    distractor_vars = []
    distractor_values = []
    for _, row in annotated.iterrows():
        vars_, values_ = extract_noop_distractor_variables(row)
        distractor_vars.append(vars_)
        distractor_values.append(values_)
    annotated["distractor_vars"] = distractor_vars
    annotated["distractor_values"] = distractor_values
    return annotated


def annotate_noop_relevant_values(df):
    """Add computation-relevant variable/value columns to a noop dataframe copy."""
    annotated = df.copy()
    relevant_vars = []
    relevant_values = []
    for _, row in annotated.iterrows():
        vars_, values_ = extract_noop_relevant_variables(row)
        relevant_vars.append(vars_)
        relevant_values.append(values_)
    annotated["relevant_vars"] = relevant_vars
    annotated["relevant_values"] = relevant_values
    return annotated


def annotate_primary_noop_distractor(df):
    """Add the canonical single distractor value from the reviewed noop clause."""
    annotated = df.copy()
    annotated["primary_distractor_value"] = annotated.apply(
        extract_primary_noop_distractor_value, axis=1
    )
    return annotated


def load_noop_source_annotations() -> pd.DataFrame:
    """Load all split noop CSVs and annotate with relevant_values and primary_distractor_value.

    Returns a DataFrame with columns: original_id, instance, corrupted_prompt,
    relevant_vars, relevant_values, primary_distractor_value.
    """
    frames = []
    for path in sorted(_NOOP_SOURCE_DIR.glob("test_gsm_noop_set_*.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = annotate_noop_relevant_values(df)
    df = annotate_primary_noop_distractor(df)
    return df[[
        "original_id", "instance", "question",
        "relevant_vars", "relevant_values", "primary_distractor_value",
    ]].rename(columns={"question": "corrupted_prompt"})
