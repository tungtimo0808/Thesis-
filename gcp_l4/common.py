"""Small shared helpers: live-streaming subprocess + environment setup + checkpoint discovery."""
import glob
import os
import subprocess
import sys

import config


def make_env():
    """Environment for every swift call: force Hugging Face (not ModelScope), fast transfer,"""
    env = os.environ.copy()
    env["USE_HF"] = "1"
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def with_max_pixels(env, uses_max_pixels):
    if uses_max_pixels:
        env = dict(env)
        env["MAX_PIXELS"] = str(config.MAX_PIXELS)
    return env


def run_streaming(command, env=None, cwd=None):
    """Run a command and print its output LIVE (line by line), so a long step is visibly alive,"""
    print(">>>", " ".join(str(c) for c in command), flush=True)
    proc = subprocess.Popen(command, env=env, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        sys.stdout.flush()
    proc.wait()
    if proc.returncode != 0:
        raise SystemExit("Command failed with exit code %d: %s" % (proc.returncode, " ".join(map(str, command))))


def find_last_checkpoint(folder):
    """Newest checkpoint-N directory in folder, or None."""
    last_path, last_step = None, -1
    for path in glob.glob(os.path.join(folder, "checkpoint-*")):
        n = os.path.basename(path).replace("checkpoint-", "")
        if n.isdigit() and int(n) > last_step:
            last_step, last_path = int(n), path
    return last_path


def list_checkpoints(folder):
    """[(step, path), ...] sorted by step."""
    pairs = []
    for path in glob.glob(os.path.join(folder, "checkpoint-*")):
        n = os.path.basename(path).replace("checkpoint-", "")
        if n.isdigit():
            pairs.append((int(n), path))
    pairs.sort()
    return pairs


def resolve_model_key(argv):
    """First CLI arg must be a model key from the registry."""
    if len(argv) < 2 or argv[1] not in config.MODELS:
        sys.exit("Usage: python %s <model_key>  (one of: %s)" %
                 (os.path.basename(argv[0]), ", ".join(config.model_keys())))
    return argv[1]
