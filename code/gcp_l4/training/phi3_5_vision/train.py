"""
Training script — Phi-3.5-vision-instruct on the PAN924 dental report task.

This is the exact QLoRA fine-tuning configuration used to train Phi-3.5-vision for the
thesis "Vision-Language Models for Automatic Dental Report Generation". One model, one
file, so the training of this model can be read and checked on its own.

Phi-3.5-vision is the smallest of the rare-loss models (about 4.2B). It uses a CLIP
ViT-L/14 vision encoder and the Phi-3.5-mini language model.

Model-specific detail: this model is loaded with the "eager" attention implementation
(attn_impl="eager"). Phi-3.5-vision's custom modeling code is not reliable with the
fused/flash attention path on this stack, so eager attention is used for stability.

What the script does, step by step:
  1. Load Phi-3.5-vision-instruct from Hugging Face (eager attention).
  2. Quantise the frozen base model to 4-bit (QLoRA, NF4 + double quant).
  3. Attach small LoRA adapters to the language-model linear layers (vision encoder frozen).
  4. Up-weight the training loss on the tokens that spell a rare disease code
     (rare-token loss, weight 3.0) to fight the strong class imbalance.
  5. Fine-tune for 2 epochs on the balanced PAN924 training split (5,817 rows) and
     save a checkpoint every 100 steps.

How to run:
    cd ~/pan924
    python gcp_l4/training/phi3_5_vision/train.py

Hardware: a single NVIDIA L4 GPU (24 GB). Framework: ms-swift, via its Python API.
"""
import os
import sys

from swift.llm import sft_main, TrainArguments


GCP_L4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GCP_L4_DIR)
from rare_token_ids import compute_rare_token_ids   # noqa: E402

MODEL = "microsoft/Phi-3.5-vision-instruct"
OUTPUT_DIR = os.path.expanduser("~/pan924_runs/phi")

RARE_CONDITION_CODES = ["Dc", "Im", "P", "Rr", "M3f", "C", "M3i", "CpuM"]
COMMON_CONDITION_CODES = ["H", "R"]
RARE_LOSS_WEIGHT = 3.0


def configure_environment():
    """Set the environment variables that swift and the rare-loss plugin read."""
    # Phi-3.5-vision uses dynamic cropping for high-resolution input; MAX_PIXELS caps it.
    os.environ["MAX_PIXELS"] = "401408"
    os.environ["USE_HF"] = "1"

    rare_token_ids, _ = compute_rare_token_ids(
        MODEL, RARE_CONDITION_CODES, COMMON_CONDITION_CODES
    )
    os.environ["RARE_TOKEN_IDS"] = ",".join(str(i) for i in rare_token_ids)
    os.environ["RARE_LOSS_WEIGHT"] = str(RARE_LOSS_WEIGHT)
    print("[rare-loss] up-weighting %d token ids x%.1f" % (len(rare_token_ids), RARE_LOSS_WEIGHT))


def build_training_arguments() -> TrainArguments:
    """All training settings, written out in full (no hidden shared config)."""
    return TrainArguments(
        # ---- model ----
        model=MODEL,
        train_type="lora",
        torch_dtype="bfloat16",
        attn_impl="eager",                 # Phi-3.5-vision specific: use eager attention

        # ---- data (image paths inside the jsonl are relative to the run dir) ----
        dataset=["vlm_report_dataset/converted/qwen/train.jsonl"],   # balanced split, 5,817 rows
        val_dataset=["vlm_report_dataset/converted/qwen/val.jsonl"],
        split_dataset_ratio=0.0,

        # ---- 4-bit QLoRA ----
        quant_method="bnb",
        quant_bits=4,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",

        # ---- LoRA ----
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["all-linear"],
        freeze_vit=True,

        # ---- optimisation schedule ----
        num_train_epochs=2,
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,    # effective batch size = 16
        dataloader_num_workers=4,
        max_length=4096,
        gradient_checkpointing=True,

        # ---- rare-token loss (the imbalance-handling plugin) ----
        external_plugins=[os.path.join(GCP_L4_DIR, "rare_loss.py")],
        loss_type="rare_weighted",

        # ---- evaluation / checkpointing ----
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=12,
        logging_steps=5,

        # ---- reproducibility / output ----
        seed=924,
        add_version=False,
        output_dir=OUTPUT_DIR,
    )


def main():
    configure_environment()
    arguments = build_training_arguments()
    print("Training Phi-3.5-vision-instruct -> %s" % OUTPUT_DIR)
    sft_main(arguments)


if __name__ == "__main__":
    main()
