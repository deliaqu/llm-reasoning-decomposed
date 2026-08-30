#!/bin/bash
# Generate PhantomWiki universes for the probe experiment (seeds 4-27).
#
# The pre-generated v1 dump (seeds 1-3, depth-20 hard mode) yields too few
# trustworthy direct-correct rows (~100-150) across too few CV groups (3
# universes) for presence probes. This generates 24 additional small universes
# in EASY mode at question-depth 8, which produces difficulty 1-3 questions
# only — the slice Llama-3.3-70B actually solves in direct mode (77.6% at
# d1-2) — with ~75 single-answer questions per universe, about half of them
# guess-proof name answers.
#
# Requires the dedicated `phantomwiki` conda env (python 3.12 + swi-prolog
# from conda-forge + pip phantom-wiki==1.0.3); does NOT touch the llm-reason env.
#
# Submit from project root (CPU-only — no #SBATCH --gres line, so submit.sh
# requests no GPU):
#   ./submit.sh data_scripts/phantomwiki/generate_universes.sh
#SBATCH -J pw_generate
#SBATCH -c 4
#SBATCH -N 1 -n 1
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH --output=data_scripts/phantomwiki/logs/%x_%A.out
#SBATCH --error=data_scripts/phantomwiki/logs/%x_%A.err
#SBATCH --open-mode=append
set -euo pipefail

source /apps/software/spack/gcc/9.3.0/anaconda3/2024.02-1-wikgcxuyjciwhcgxqkpggiqlsqe3dt4a/etc/profile.d/conda.sh
conda activate phantomwiki

# Under sbatch the script runs from the slurmd spool copy, so BASH_SOURCE
# does not point into the repo — prefer SLURM_SUBMIT_DIR (project root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
OUT_ROOT="$PROJECT_ROOT/data/phantomwiki/generated"
mkdir -p "$OUT_ROOT"

for seed in $(seq 4 27); do
  out="$OUT_ROOT/seed_${seed}"
  if [ -f "$out/questions.json" ]; then
    echo "seed $seed: exists, skipping"
    continue
  fi
  phantom-wiki-generate \
    --seed "$seed" \
    --num-family-trees 2 --max-family-tree-size 25 \
    --question-depth 8 --easy-mode \
    --num-questions-per-type 10 \
    --article-format json --question-format json \
    --use-multithreading --quiet \
    --output-dir "$out"
  n=$(python3 -c "import json; print(len(json.load(open('$out/questions.json'))))")
  echo "seed $seed: generated ($n questions)"
done
echo "Done: $(ls -d "$OUT_ROOT"/seed_* | wc -l) universes in $OUT_ROOT"
