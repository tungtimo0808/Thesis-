"""
Training script — Qwen2.5-VL-3B-Instruct on the PAN924 dental report task.

This is the exact QLoRA fine-tuning configuration used to train Qwen2.5-VL-3B for the
thesis "Vision-Language Models for Automatic Dental Report Generation". One model, one
file, so the training of this model can be read and checked on its own.

Why this model: it is the smaller version of the same Qwen2.5-VL family as the 7B model.
Training both sizes with the same recipe lets us measure the effect of model size on its
own, with the architecture held fixed.

Note on the loss: this model was trained with the standard token-level cross-entropy loss.
The rare-token loss up-weighting (used for Qwen-7B, InternVL3 and Phi) was NOT applied here.

What the script does, step by step:
  1. Load Qwen2.5-VL-3B-Instruct from Hugging Face.
  2. Quantise the frozen base model to 4-bit (QLoRA, NF4 + double quant).
  3. Attach small LoRA adapters to the language-model linear layers (vision encoder frozen).
  4. Fine-tune for 2 epochs on the balanced PAN924 training split (5,817 rows) and
     save a checkpoint every 100 steps.

How to run:
    cd ~/pan924
    python gcp_l4/training/qwen2_5_vl_3b/train.py

Hardware: a single NVIDIA L4 GPU (24 GB). Framework: ms-swift, via its Python API.
"""
import os

from swift.llm import sft_main, TrainArguments


MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR = os.path.expanduser("~/pan924_runs/qwen3b")


def configure_environment():
    """Set the environment variables that swift reads."""
    # Qwen2.5-VL keeps the native image resolution, capped by MAX_PIXELS.
    os.environ["MAX_PIXELS"] = "401408"
    # Download weights from Hugging Face, not ModelScope.
    os.environ["USE_HF"] = "1"


def build_training_arguments() -> TrainArguments:
    """All training settings, written out in full (no hidden shared config)."""
    return TrainArguments(
        # ---- model ----
        model=MODEL,
        train_type="lora",
        torch_dtype="bfloat16",

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
    print("Training Qwen2.5-VL-3B-Instruct -> %s" % OUTPUT_DIR)
    sft_main(arguments)


if __name__ == "__main__":
    main()
