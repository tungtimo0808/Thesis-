"""Full pipeline for ONE model: train -> select checkpoint -> evaluate -> visualize.

Runs the four steps in order. Run it inside tmux so it survives an SSH disconnect.

Usage:
  python run_model.py qwen                 # plain
  python run_model.py qwen --rare-loss     # extra flags are forwarded to train.py only

Models: qwen | qwen3b | internvl | phi | paligemma
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run_step(script_and_args):
    """Run one pipeline step (a python script) and stop the pipeline if it fails."""
    print("\n########## " + " ".join(script_and_args) + " ##########", flush=True)
    subprocess.run([sys.executable] + script_and_args, cwd=HERE, check=True)


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python run_model.py <qwen|qwen3b|internvl|phi|paligemma> [train flags]")
    model = sys.argv[1]
    train_flags = sys.argv[2:]          # forwarded to train.py only (e.g. --rare-loss)

    run_step(["train.py", model] + train_flags)
    run_step(["select_checkpoint.py", model])
    run_step(["evaluate.py", model])
    run_step(["visualize.py", model])

    print("\n########## DONE %s -- results in ~/pan924_runs/%s/ ##########" % (model, model))


if __name__ == "__main__":
    main()
