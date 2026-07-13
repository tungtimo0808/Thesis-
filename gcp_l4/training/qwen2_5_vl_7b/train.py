"""Training script — Qwen2.5-VL-7B-Instruct on the PAN924 dental report task."""
import os
import sys

from swift.llm import sft_main, TrainArguments


# ---------------------------------------------------------------------------
GCP_L4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GCP_L4_DIR)
from rare_token_ids import compute_rare_token_ids   # noqa: E402

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR = os.path.expanduser("~/pan924_runs/qwen")

# Rare disease codes that the loss should pay extra attention to, and the two
RARE_CONDITION_CODES = ["Dc", "Im", "P", "Rr", "M3f"]
COMMON_CONDITION_CODES = ["H", "R"]
RARE_LOSS_WEIGHT = 3.0


def configure_environment():
    """Set the environment variables that swift and the rare-loss plugin read."""
    # Qwen2.5-VL keeps the native image resolution, capped by MAX_PIXELS. A high
    os.environ["MAX_PIXELS"] = "401408"
    # Download weights from Hugging Face, not ModelScope.
    os.environ["USE_HF"] = "1"

    # The rare-token loss plugin (rare_loss.py) reads these two variables. We
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
        train_type="lora",                 # LoRA adapters, not full fine-tuning
        torch_dtype="bfloat16",

        # ---- data (image paths inside the jsonl are relative to the run dir) ----
        dataset=["vlm_report_dataset/converted/qwen/train.jsonl"],   # balanced split, 5,817 rows
        val_dataset=["vlm_report_dataset/converted/qwen/val.jsonl"],
        split_dataset_ratio=0.0,           # do not carve a val set out of train; use val.jsonl

        # ---- 4-bit QLoRA (lets a 7-8B model fit on a 24 GB GPU) ----
        quant_method="bnb",
        quant_bits=4,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",

        # ---- LoRA ----
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["all-linear"],     # LoRA on every linear layer of the LLM
        freeze_vit=True,                   # keep the pretrained vision encoder frozen

        # ---- optimisation schedule ----
        num_train_epochs=2,
        learning_rate=1e-4,
        weight_decay=0.1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=1,     # one sample at a time (7B + image is heavy)
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,    # effective batch size = 1 x 16 = 16
        dataloader_num_workers=4,
        max_length=4096,
        gradient_checkpointing=True,       # trade compute for memory

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
    print("Training Qwen2.5-VL-7B-Instruct -> %s" % OUTPUT_DIR)
    sft_main(arguments)


if __name__ == "__main__":
    main()
