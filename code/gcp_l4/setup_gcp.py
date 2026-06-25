"""One-shot setup on a fresh GCP L4 VM: unzip the data + code, install dependencies, verify the GPU.

This file is SELF-CONTAINED (imports nothing local) so you can run it before anything is unzipped.

WHAT TO UPLOAD to the VM (any folder, e.g. ~):
  - pan924_vlm.zip        (the dataset: ~2.3 GB — jsonl + 4620 images + eval scripts)
  - pan924_gcp_l4.zip     (this code: the gcp_l4/ scripts)
  - setup_gcp.py          (this file)

THEN:
  python3 setup_gcp.py
  # ... it unzips both into ~/pan924, pip-installs, prints torch/GPU info, builds the balanced data.

RESULT layout:
  ~/pan924/vlm_report_dataset/...     (data)
  ~/pan924/gcp_l4/*.py                (code)
  ~/pan924_runs/<model>/              (checkpoints + results, created during training)

Flags:
  --data-zip PATH   (default ./pan924_vlm.zip)
  --code-zip PATH   (default ./pan924_gcp_l4.zip)
  --data-root PATH  (default ~/pan924)
  --skip-install    (don't pip install)
  --skip-oversample (don't build train_balanced.jsonl)
"""
import argparse
import os
import subprocess
import sys
import zipfile

PKGS = [
    "ms-swift>=3.2,<4.0", "accelerate", "bitsandbytes", "peft",
    "qwen_vl_utils", "timm", "einops", "sentencepiece", "hf_transfer",
    "matplotlib", "pandas", "numpy", "Pillow",
]


def unzip(zip_path, dest, marker):
    """Extract zip_path into dest unless `marker` (a path under dest) already exists."""
    if os.path.exists(os.path.join(dest, marker)):
        print("[skip] %s already present in %s" % (marker, dest))
        return
    if not os.path.exists(zip_path):
        print("[WARN] %s not found — skipping. Upload it and re-run if you need it." % zip_path)
        return
    print("[unzip] %s -> %s" % (zip_path, dest))
    os.makedirs(dest, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print("[ok] extracted %s" % os.path.basename(zip_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-zip", default="pan924_vlm.zip")
    ap.add_argument("--code-zip", default="pan924_gcp_l4.zip")
    ap.add_argument("--data-root", default=os.path.expanduser("~/pan924"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-oversample", action="store_true")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    print("DATA_ROOT =", data_root)

    # 1) unzip data + code into DATA_ROOT
    unzip(args.data_zip, data_root, os.path.join("vlm_report_dataset", "converted", "qwen", "train.jsonl"))
    unzip(args.code_zip, data_root, os.path.join("gcp_l4", "train.py"))

    # 2) install dependencies
    if not args.skip_install:
        print("\n[pip] installing dependencies (this can take a few minutes)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + PKGS, check=True)
    else:
        print("[skip] pip install")

    # 3) verify GPU + versions
    print("\n[verify] importing torch + swift ...")
    os.environ["USE_HF"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    try:
        import torch
        print("  torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("  GPU:", torch.cuda.get_device_name(0),
                  "| bf16:", torch.cuda.is_bf16_supported(),
                  "| VRAM: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
        else:
            print("  [WARN] CUDA not available to torch. If nvidia-smi works, your torch build's CUDA")
            print("         version may not match the driver — reinstall torch for this CUDA, e.g.:")
            print("         pip install torch --index-url https://download.pytorch.org/whl/cu124")
        import bitsandbytes  # noqa: F401
        print("  bitsandbytes OK (needed for 4-bit QLoRA)")
        import swift
        print("  ms-swift", swift.__version__)
    except Exception as e:
        print("  [WARN] verification import failed:", e)
        print("  Fix the environment before training (see README troubleshooting).")

    # 4) build the rare-class-balanced training file.
    # Prefer augment_rare.py (image augmentation = better generalisation than plain duplication);
    # fall back to oversample.py if Pillow / images are unavailable.
    code_dir = os.path.join(data_root, "gcp_l4")
    if not args.skip_oversample:
        env = os.environ.copy()
        env["PAN924_DATA_ROOT"] = data_root
        builder = "augment_rare.py" if os.path.exists(os.path.join(code_dir, "augment_rare.py")) else "oversample.py"
        print("\n[rebalance] building train_balanced.jsonl via %s ..." % builder)
        rc = subprocess.run([sys.executable, builder], cwd=code_dir, env=env).returncode
        if rc != 0 and builder == "augment_rare.py":
            print("[rebalance] augmentation failed (Pillow?) — falling back to plain oversample.py")
            subprocess.run([sys.executable, "oversample.py"], cwd=code_dir, env=env, check=False)

    print("\n=== SETUP DONE ===")
    print("Next steps:")
    print("  cd %s" % code_dir)
    print("  python train.py qwen --smoke      # ~2 min: prove the config works")
    print("  python train.py qwen              # full train (run inside tmux — see README)")


if __name__ == "__main__":
    main()
