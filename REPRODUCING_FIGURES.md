# Reproducing the figures and tables

This table maps each figure and table in the paper to the python command that
produces it. All commands assume the working directory is this
bundle root and a Llama-3.3-70B-Instruct checkpoint is loadable through
Hugging Face. Swap `--model_id` for the appendix robustness models
(`google/gemma-2-9b-it`, `Qwen/Qwen2.5-14B-Instruct`).


| Paper artifact | Command |
|---|---|
| Figure 1 — pipeline overview (TikZ, inline) | inline LaTeX (no data) |
| Figure 2a — template cosine (Stage 1) | `python run_template_similarity.py --mode direct --correct_only` |
| Figure 2b — presence probe (Stages 1, 3, 4) | `python run_presence_probe.py --dataset gsm_symbolic --mode direct --correct_only` |
| Figure 3 — Stage-2 patching schematic | `python plot_p1_stage2_method.py` (schematic, no data) |
| Figure 4a — question-span patching (Stage 2) | `python run_cot_swap_activation_patching.py --contrast p1_vs_padded_symbolic` |
| Figure 4b — commit-readiness (Stage 2) | Run once per scope: `python run_cot_swap_activation_patching.py --contrast p1_vs_padded_symbolic --cot_source symbolic_aligned --position cot_end --scope layer`; repeat with `--scope mlp` and `--scope attn_output` |
| Figure 5a — cot_boundary value-tokens (Stage 3) | `python run_noop_activation_patching.py --experiment cot_boundary --pair_dataset gsm_sym_within_template --patch_positions value_tokens --scope layer` |
| Figure 5b — direct prompt-end decomposition (Stage 3) | Run once per scope: `python run_noop_activation_patching.py --experiment direct --pair_dataset gsm_sym_within_template --patch_positions prompt_end --scope layer`; repeat with `--scope mlp` and `--scope attn_output` |
| Figure 6 — NoOp fragility (Stage 2 break test) | `python run_noop_activation_patching.py --experiment cot_boundary --pair_dataset filler_df_vs_noop_clean_tfm` |
| Figure 7 — engagement-anchored DLA schematic | `python plot_dla_method.py` (schematic, no data) |
| Figure 19 — SVAMP four-stage signatures (appendix) | `python data_scripts/svamp/build_svamp_probe_metadata.py` → `python run_svamp_presence_probe.py --mode direct --correct_only`; operand and answer rows use `run_svamp_variants_presence_probe.py` and `diag_svamp_answer_within_problem.py` |
| Tables 1 / 2 — head ablation + amplification | `python run_cot_swap_dla.py --contrast filler_df_correct_vs_noop_clean_wrong` → `python run_cot_swap_head_scaling.py --contrast filler_df_correct_vs_noop_clean_wrong --side noop --patch_position cot_end` |
| Table 3 — failure classification under ablation | Roll-outs: `python run_cot_swap_head_scaling.py --contrast filler_df_correct_vs_noop_clean_wrong --side noop --scale 0`; the blind annotation sheets and adjudicated labels are in `error_typing_review/` |
| PhantomWiki probes (appendix, in text) | `bash data_scripts/phantomwiki/generate_universes.sh` → `python data_scripts/phantomwiki/build_phantomwiki_dataset.py` → `python run_phantomwiki_role_probe.py`; the reported chain-membership value is the difficulty-matched contrast, `python diag_role_probe_adjacency_matched_dm.py`, with `diag_role_probe_renamed_negatives.py` as the copy-identity control |

`--model_id meta-llama/Llama-3.3-70B-Instruct` is the default. All `run_*` scripts
accept `--help` for their full argument surface; `--plot_only` regenerates
figures from cached `.jsonl`/`.npy` outputs without re-running model forward
passes.

## Pipeline order

1. **Stage 1** (Fig 1): `run_template_similarity.py` and `run_presence_probe.py`
2. **Stage 2** (Fig 3): `run_cot_swap_activation_patching.py`
3. **Stages 3–4** (Fig 4): `run_noop_activation_patching.py` with
   `--pair_dataset gsm_sym_within_template`
4. **NoOp fragility** (Fig 5): `run_noop_activation_patching.py` with
   `--pair_dataset filler_df_vs_noop_clean_tfm`
5. **Attention-head mechanism** (Tables 1, 2): `run_cot_swap_dla.py` →
   `run_cot_swap_head_scaling.py`
6. **Failure classification** (Table 3): re-run head scaling at `--scale 0`, then
   label traces blind; see `error_typing_review/`
7. **Generalization** (appendix): build the datasets under `data_scripts/`, then
   `run_svamp_presence_probe.py` / `run_phantomwiki_role_probe.py` and the
   matching `diag_*` contrasts
8. **Render PDFs**: `python paper_render.py`

## Notes

- For 70B activations the hidden-state cache is large (~100 GB per mode for
  paired datasets). Override `CACHE_DIR` via the env var:
  `CACHE_DIR=/big/disk python run_template_similarity.py ...`.
- The interpretability scripts expect inference outputs (model generations +
  behavioural-correctness labels) to live at
  `<repo>/results/disentangled_evaluation/transformers_direct/...`. Generate them
  with `run_inference_transformers_direct.py`, which writes both the generations
  and the hidden-state cache the probes read.
