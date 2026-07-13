#!/usr/bin/env bash
# Full pipeline for ONE model: train -> select checkpoint -> evaluate -> visualize.
# Run inside tmux so it survives SSH disconnects:
#   tmux new -s pan924
#   bash run_model.sh qwen                 # plain
#   bash run_model.sh qwen --rare-loss     # with rare-token loss up-weighting (extra args go to TRAIN only)
set -euo pipefail
MODEL="${1:?Usage: bash run_model.sh <qwen|internvl|llava|phi|paligemma> [train flags e.g. --rare-loss]}"
shift                              # remaining args ("$@") are forwarded to train.py only
cd "$(dirname "$0")"

echo "########## TRAIN $MODEL $* ##########"
python train.py "$MODEL" "$@"
echo "########## SELECT CHECKPOINT $MODEL ##########"
python select_checkpoint.py "$MODEL"
echo "########## EVALUATE $MODEL ##########"
python evaluate.py "$MODEL"
echo "########## VISUALIZE $MODEL ##########"
python visualize.py "$MODEL"
echo "########## DONE $MODEL — results in ~/pan924_runs/$MODEL/ ##########"
