"""Training script — InternVL3-8B on the PAN924 dental report task."""
import os
import sys

from swift.llm import sft_main, TrainArguments


GCP_L4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GCP_L4_DIR)
from rare_token_ids import compute_rare_token_ids   # noqa: E402

MODEL = "OpenGVLab/InternVL3-8B"
OUTPUT_DIR = os.path.expanduser("~/pan924_runs/internvl")

RARE_CONDITION_CODES = ["Dc", "Im", "P", "Rr", "M3f"]
COMMON_CONDITION_CODES = ["H", "R"]
RARE_LOSS_WEIGHT = 3.0


def configure_environment():
    """Set the environment variables that swift and the rare-loss plugin read."""
    # InternVL3 cuts the image into 448px tiles; MAX_PIXELS caps the total size.
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
    print("Training InternVL3-8B -> %s" % OUTPUT_DIR)
    sft_main(arguments)


if __name__ == "__main__":
    main()
