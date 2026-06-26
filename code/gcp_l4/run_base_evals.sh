#!/usr/bin/env bash
# Run the zero-shot BASE-model evaluation (no LoRA adapter) for one or more models,
# one after another. They run sequentially so they never compete for the single GPU.
#
# Usage:
#   bash run_base_evals.sh internvl qwen        # base eval for InternVL3, then Qwen-7B
#   bash run_base_evals.sh qwen                  # just one model
#
# Each model writes its results to ~/pan924_runs/<model>/{metrics,report}_<model>_base.*
# and a log to ~/pan924_runs/<model>/eval_base.log
set -euo pipefail
cd "$(dirname "$0")"

for MODEL in "$@"; do
    echo "########## BASE EVAL: ${MODEL} ##########"
    python evaluate_base.py "${MODEL}" 2>&1 | tee "${HOME}/pan924_runs/${MODEL}/eval_base.log"
done

echo "########## DONE base eval: $* ##########"
