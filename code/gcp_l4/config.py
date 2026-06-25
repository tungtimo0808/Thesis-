"""Shared configuration for the PAN924 VLM training pipeline on a GCP L4 (24 GB) VM.

This is the SINGLE source of settings — every other script imports from here, so the 5 models
train under identical conditions (a fair comparison) and there are no hand-tuned magic numbers
scattered around.

WHY THIS DIFFERS FROM THE A100 NOTEBOOKS (and why it's still "correct", not "linh tinh"):
  - The A100 path used bf16 LoRA. A 7-8B model in bf16 needs ~28-40 GB and will NOT fit the L4's
    24 GB. So here we use 4-bit QLoRA (NF4 + double quant, bf16 compute). The weights drop to
    ~5 GB, leaving plenty of VRAM for activations + a high image resolution. Quality loss vs bf16
    LoRA is small (~1-2 pts), and crucially: ALL 5 models use the SAME 4-bit config, so they stay
    comparable to each other.
  - Everything else (LoRA rank/alpha, lr, effective batch 16, eval/save every 100) is identical
    to the A100 run.

RARE-CLASS HELP (the dataset is very imbalanced: H:Dc ~ 47:1):
  - USE_BALANCED_TRAIN oversamples reports that contain rare conditions (see oversample.py), which
    is the single most effective lever to make rare diseases "easier to recognise".
  - MAX_PIXELS is kept HIGH (small lesions like caries Dc / residual root Rr need detail). 4-bit
    weights free the VRAM to afford it.
"""
import os

# --------------------------------------------------------------------------- paths (all LOCAL — no Google Drive)
# DATA_ROOT must contain `vlm_report_dataset/` (that's how pan924_vlm.zip extracts). The image paths
# inside the jsonl are relative to DATA_ROOT, so every script chdir()s here before calling swift.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("PAN924_DATA_ROOT", os.path.expanduser("~/pan924"))
OUTPUT_ROOT = os.environ.get("PAN924_OUTPUT_ROOT", os.path.expanduser("~/pan924_runs"))

DATASET_DIR = "vlm_report_dataset/converted/qwen"      # relative to DATA_ROOT (generic messages format; swift re-templates per model)
EVAL_SCRIPT = "vlm_report_dataset/scripts/eval_report.py"
ADAPTER_SCRIPT = "vlm_report_dataset/training/tools/swift_pred_to_eval.py"

TRAIN_JSONL = DATASET_DIR + "/train.jsonl"
TRAIN_BALANCED_JSONL = DATASET_DIR + "/train_balanced.jsonl"   # made by oversample.py
VAL_JSONL = DATASET_DIR + "/val.jsonl"
TEST_JSONL = DATASET_DIR + "/test.jsonl"

# --------------------------------------------------------------------------- model registry
# key -> dict(model=HF id, uses_max_pixels, gated, attn_impl)
MODELS = {
    "qwen":      {"model": "Qwen/Qwen2.5-VL-7B-Instruct",             "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "qwen3b":    {"model": "Qwen/Qwen2.5-VL-3B-Instruct",             "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "internvl":  {"model": "OpenGVLab/InternVL3-8B",                  "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "llava":     {"model": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "phi":       {"model": "microsoft/Phi-3.5-vision-instruct",       "uses_max_pixels": True,  "gated": False, "attn_impl": "eager"},
    "paligemma": {"model": "google/paligemma2-3b-pt-448",             "uses_max_pixels": False, "gated": True,  "attn_impl": None},
}

# --------------------------------------------------------------------------- training hyper-parameters (shared)
# LoRA — identical to the A100 run.
TRAIN_TYPE = "lora"
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = "all-linear"
FREEZE_VIT = "true"

# 4-bit QLoRA — the one necessary change for the 24 GB L4.
QUANT_METHOD = "bnb"
QUANT_BITS = 4
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = "true"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
TORCH_DTYPE = "bfloat16"

# schedule / optimisation — identical to the A100 run.
NUM_EPOCHS = 2
LEARNING_RATE = "1e-4"
WEIGHT_DECAY = "0.1"
WARMUP_RATIO = "0.05"
LR_SCHEDULER = "cosine"
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16          # effective batch = 1 * 16 = 16
MAX_LENGTH = 4096
                              # InternVL/LLaVA tile images into MORE tokens — if the --smoke run prints a
                              # "max_length ... truncated" warning, raise this (6144/8192) or lower MAX_PIXELS.
GRAD_CHECKPOINTING = "true"
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 12
SEED = 924

# Image resolution. 1003520 = 1280*28*28 (same as A100). 4-bit weights leave room for this on the
# L4, and high resolution is what lets the model SEE small rare lesions. If ANY model OOMs at step
# 0, lower this (1003520 -> 802816 -> 602112) — OOM happens immediately, so it costs ~nothing to test.
MAX_PIXELS = 401408

# --------------------------------------------------------------------------- rare-class handling
USE_BALANCED_TRAIN = False
# oversample.py targets each condition up to this many tooth-occurrences (rare ones get duplicated),
# capping any single report's replication at REPLICATION_CAP so common reports aren't blown up.
OVERSAMPLE_TARGET = 3000
REPLICATION_CAP = 6

# --------------------------------------------------------------------------- experimental: rare-token loss up-weighting
# Opt-in via `python train.py <model> --rare-loss`. Multiplies the per-token cross-entropy by
# RARE_LOSS_WEIGHT for label positions that are tokens of a rare disease code — directly countering
# the "relabel rare -> H/R" bias WITHOUT dragging H/R along (unlike oversampling). Implemented as an
# ms-swift external-plugin loss (rare_loss.py); validate it with --smoke (the plugin API can vary by
# swift version). RARE_CODES matches eval_report.py's RARE set.
RARE_CODES = ["Dc", "Im", "P", "Rr", "M3f", "C", "M3i", "CpuM"]
COMMON_CODES = ["H", "R"]      # used to subtract shared punctuation ids when isolating rare tokens
RARE_LOSS_WEIGHT = 3.0

# --------------------------------------------------------------------------- inference / eval
INFER_BACKEND = "pt"           # reliable everywhere. vLLM (~10x faster) is optional — see README.
INFER_MAX_BATCH_SIZE = 8       # lower to 4 if eval OOMs
INFER_MAX_NEW_TOKENS = 1280    # longest gold report = 1020 tokens; 1280 = safe headroom (it's a cap, not a fixed cost)


def model_keys():
    return list(MODELS.keys())


def output_dir(model_key):
    return os.path.join(OUTPUT_ROOT, model_key)


def train_dataset_path():
    """The balanced file if enabled AND present, else the original train.jsonl."""
    if USE_BALANCED_TRAIN and os.path.exists(os.path.join(DATA_ROOT, TRAIN_BALANCED_JSONL)):
        return TRAIN_BALANCED_JSONL
    return TRAIN_JSONL
