import os

# Paths for local server data and outputs.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("PAN924_DATA_ROOT", os.path.expanduser("~/pan924"))
OUTPUT_ROOT = os.environ.get("PAN924_OUTPUT_ROOT", os.path.expanduser("~/pan924_runs"))

DATASET_DIR = "vlm_report_dataset/converted/qwen"
EVAL_SCRIPT = "vlm_report_dataset/scripts/eval_report.py"
ADAPTER_SCRIPT = "vlm_report_dataset/training/tools/swift_pred_to_eval.py"

TRAIN_JSONL = DATASET_DIR + "/train.jsonl"
TRAIN_BALANCED_JSONL = DATASET_DIR + "/train_balanced.jsonl"
VAL_JSONL = DATASET_DIR + "/val.jsonl"
TEST_JSONL = DATASET_DIR + "/test.jsonl"

# Model registry.
MODELS = {
    "qwen":      {"model": "Qwen/Qwen2.5-VL-7B-Instruct",             "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "qwen3b":    {"model": "Qwen/Qwen2.5-VL-3B-Instruct",             "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "internvl":  {"model": "OpenGVLab/InternVL3-8B",                  "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "llava":     {"model": "llava-hf/llava-onevision-qwen2-7b-ov-hf", "uses_max_pixels": True,  "gated": False, "attn_impl": None},
    "phi":       {"model": "microsoft/Phi-3.5-vision-instruct",       "uses_max_pixels": True,  "gated": False, "attn_impl": "eager"},
    "paligemma": {"model": "google/paligemma2-3b-pt-448",             "uses_max_pixels": False, "gated": True,  "attn_impl": None},
}

# Shared LoRA hyperparameters.
TRAIN_TYPE = "lora"
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
TARGET_MODULES = "all-linear"
FREEZE_VIT = "true"

# 4-bit QLoRA settings for the 24 GB L4 GPU.
QUANT_METHOD = "bnb"
QUANT_BITS = 4
BNB_4BIT_QUANT_TYPE = "nf4"
BNB_4BIT_USE_DOUBLE_QUANT = "true"
BNB_4BIT_COMPUTE_DTYPE = "bfloat16"
TORCH_DTYPE = "bfloat16"

# Training schedule and optimization.
NUM_EPOCHS = 2
LEARNING_RATE = "1e-4"
WEIGHT_DECAY = "0.1"
WARMUP_RATIO = "0.05"
LR_SCHEDULER = "cosine"
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16
MAX_LENGTH = 4096
GRAD_CHECKPOINTING = "true"
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 12
SEED = 924

MAX_PIXELS = 401408

# Rare-class oversampling settings.
USE_BALANCED_TRAIN = False
OVERSAMPLE_TARGET = 3000
REPLICATION_CAP = 6

# Rare-token weighted loss settings.
RARE_CODES = ["Dc", "Im", "P", "Rr", "M3f"]
COMMON_CODES = ["H", "R"]
RARE_LOSS_WEIGHT = 3.0

# Inference and evaluation settings.
INFER_BACKEND = "pt"
INFER_MAX_BATCH_SIZE = 8
INFER_MAX_NEW_TOKENS = 1280


def model_keys():
    return list(MODELS.keys())


def output_dir(model_key):
    return os.path.join(OUTPUT_ROOT, model_key)


def train_dataset_path():
    """Return the balanced train file when enabled, otherwise the original train file."""
    if USE_BALANCED_TRAIN and os.path.exists(os.path.join(DATA_ROOT, TRAIN_BALANCED_JSONL)):
        return TRAIN_BALANCED_JSONL
    return TRAIN_JSONL
