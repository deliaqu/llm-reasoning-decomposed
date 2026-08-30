# A Four-Stage Decomposition of Word-Problem Solving and Mechanistic Fragility in LLM Math Reasoning

Code accompanying the paper by
[Zhongdi Qu](mailto:zq84@cornell.edu) and Carla P. Gomes (Cornell University).

<!-- TODO(camera-ready): replace with the final venue and paper/anthology link once the
     commitment decision is in. -->
*Venue: to appear.*

The paper proposes a four-stage decomposition of grade-school math word-problem
solving in LLMs — **Schema Abstraction**, **Operation Planning**, **Operand
Binding**, **Computation** — and uses the same scaffold to localize
NoOp-distractor fragility to the Operation Planning stage, implemented by a
small set of attention heads.

This repository contains the mechanistic-interpretability code behind every
figure and table in the paper, the dataset-construction pipelines, and the
annotation artifacts for the failure-classification experiment.

## Layout

```
.
├── README.md
├── LICENSE
├── REPRODUCING_FIGURES.md                # figure/table → command mapping
├── requirements.txt
├── config.py                             # paths, dataset metadata, model id → short-name map
├── paths.py                              # path constants (DATA_DIR, RESULT_DIR, CACHE_DIR)
├── instructions.py                       # prompt templates + TASK_CONFIG dataset registry
├── extract_hidden_states.py              # residual-stream activation cache
│
│   # Stage identification (Section 5)
├── run_template_similarity.py            # residual cosine, stage boundaries (Fig 1a)
├── run_presence_probe.py                 # entity/operator/number/answer probes (Fig 1b)
├── run_cot_swap_activation_patching.py   # Stage 2 cross-prompt patching (Fig 3)
├── run_within_template_cot_prompt_end_patching.py  # Stage 3 (Fig 4)
├── run_noop_activation_patching.py       # within-template + NoOp patching (Fig 4, 5)
│
│   # NoOp diagnosis (Section 6)
├── run_cot_swap_dla.py                   # engagement-anchored DLA per head
├── run_cot_swap_dla_divergence.py        # divergence-anchored DLA helper
├── run_cot_swap_head_scaling.py          # head ablation / amplification (Tables 2, 3)
├── error_typing_review/                  # blind failure-classification sheets (Table 4)
│
│   # Generalization to other datasets and tasks (Appendix)
├── run_svamp_presence_probe.py           # SVAMP four-stage signatures (Table 10)
├── run_svamp_variants_presence_probe.py  # operand-resampled within-problem contrasts
├── diag_svamp_answer_within_problem.py   # within-problem answer contrast
├── run_phantomwiki_presence_probe.py     # PhantomWiki question-level probes
├── run_phantomwiki_role_probe.py         # relation / chain-membership / answer probes
├── diag_role_probe_adjacency_matched.py     # matched chain-membership contrast
├── diag_role_probe_adjacency_matched_dm.py  #   + difficulty matching (reported value)
├── diag_role_probe_renamed_negatives.py     # copy-identity shortcut control
│
│   # Inference and evaluation
├── run_inference_transformers_direct.py  # generation + hidden-state capture
├── run_evaluation.py, evaluation_utils.py
│
│   # Supporting / import-only
├── run_cot_swap_head_patching.py, run_cot_swap_logit_lens.py,
├── run_attention_analysis.py, run_noop_dla.py, run_noop_target_logit_lens.py,
├── run_input_recovery.py, run_op_multiset_probe.py
│
├── paper_render.py                       # canonical figure renderer
├── data_scripts/                         # dataset construction (SVAMP, PhantomWiki)
├── data/                                 # datasets (see data/README.md)
└── utils/                                # shared helpers (hooks, logit lens, plotting)
```

The scripts are plain Python CLIs; invoke each directly (`python run_X.py ...`).
The 70B experiments need ≥160 GB of GPU memory in total (e.g. 2× A100 80 GB or
1× H200). Every `run_*` script accepts `--help`, and `--plot_only` re-renders
figures from cached `.jsonl`/`.npy` outputs without re-running forward passes.

## Datasets

Datasets are not included in this release yet; see `data/README.md` for the
expected layout and provenance. All GSM-family datasets derive from the public
GSM8K and GSM-Symbolic releases, and their construction pipelines are specified
in the paper's appendix.

The four interpretability datasets used in the main paper:

- `gsm_symbolic` — narrated GSM-Symbolic problems (clean)
- `gsm_symbolic_padded_to_p1` — length-padded variant aligned with `gsm_p1`
- `gsm_noop_clean` — NoOp distractor variant (irrelevant clause inserted)
- `gsm_filler` — length-matched filler control

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
# Optional: put the (large) activation cache somewhere with space
export CACHE_DIR=/path/to/large/cache
```

The pinned environment targets Python 3.11+ with a CUDA-capable PyTorch build.
Each script accepts `--model_id` (a Hugging Face id). The main paper reports
Llama-3.3-70B-Instruct; appendix robustness results use `google/gemma-2-9b-it`
and `Qwen/Qwen2.5-14B-Instruct`.

## Reproducing figures and tables

See `REPRODUCING_FIGURES.md` for the per-artifact command table.

## Acknowledgements

Parts of the evaluation scaffolding in `config.py` and `instructions.py` —
the disentangled prompt conditions and dataset registry — derive from the code
released with [Cheng et al. (2025)](https://aclanthology.org/2025.emnlp-main.723/),
*Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling
Abstract Formulation From Arithmetic Computation*
([repository](https://github.com/ziling-cheng/Disentangle-Math-Reasoning)),
which our four-stage account builds on. We thank the authors for releasing it.

## Citation

<!-- TODO(camera-ready): update once the paper appears in the ACL Anthology. -->
```bibtex
@inproceedings{qu2026fourstage,
  title     = {A Four-Stage Decomposition of Word-Problem Solving and Mechanistic
               Fragility in {LLM} Math Reasoning},
  author    = {Qu, Zhongdi and Gomes, Carla P.},
  year      = {2026}
}
```
