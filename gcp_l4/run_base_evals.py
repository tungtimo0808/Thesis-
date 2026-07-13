"""Run the zero-shot BASE-model evaluation (no LoRA adapter) for one or more models."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.expanduser("~/pan924_runs")


def run_base_eval(model):
    """Run evaluate_base.py for one model, streaming output to console and a log file."""
    log_dir = os.path.join(RUNS, model)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval_base.log")
    print("\n########## BASE EVAL: %s ##########" % model, flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "evaluate_base.py", model], cwd=HERE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        proc.wait()
    if proc.returncode != 0:
        sys.exit("Base eval failed for %s (exit code %d)" % (model, proc.returncode))


def main():
    models = sys.argv[1:]
    if not models:
        sys.exit("Usage: python run_base_evals.py <model> [<model> ...]")
    for model in models:
        run_base_eval(model)
    print("\n########## DONE base eval: %s ##########" % " ".join(models))


if __name__ == "__main__":
    main()
