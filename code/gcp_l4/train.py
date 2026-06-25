"""Train one model with 4-bit QLoRA on the L4. Streams logs live; auto-resumes from the last
checkpoint so an SSH disconnect / VM restart costs nothing (run it under tmux — see README).

Usage:
  python train.py qwen            # full run (resumes if checkpoints exist)
  python train.py qwen --smoke    # 5-step dry run (~2 min): proves the config works BEFORE you
                                   #   commit to hours of compute. ALWAYS run this first on a new model.

Models: qwen | internvl | llava | phi | paligemma
"""
import os
import sys

import config
from common import make_env, with_max_pixels, run_streaming, find_last_checkpoint, resolve_model_key


def build_command(model_key, smoke, rare_loss):
    m = config.MODELS[model_key]
    out_dir = config.output_dir(model_key)
    os.makedirs(out_dir, exist_ok=True)

    dataset = config.train_dataset_path()
    if smoke:
        # In smoke mode use the small (unbalanced) train file so loading is quick.
        dataset = config.TRAIN_JSONL

    cmd = [
        "swift", "sft",
        "--model", m["model"],
        "--dataset", dataset,
        "--val_dataset", config.VAL_JSONL,
        "--split_dataset_ratio", "0",
        "--train_type", config.TRAIN_TYPE,
        "--torch_dtype", config.TORCH_DTYPE,
        # ---- 4-bit QLoRA (the change that makes a 7-8B model fit the L4's 24 GB) ----
        "--quant_method", config.QUANT_METHOD,
        "--quant_bits", str(config.QUANT_BITS),
        "--bnb_4bit_quant_type", config.BNB_4BIT_QUANT_TYPE,
        "--bnb_4bit_use_double_quant", config.BNB_4BIT_USE_DOUBLE_QUANT,
        "--bnb_4bit_compute_dtype", config.BNB_4BIT_COMPUTE_DTYPE,
        # ---- LoRA (identical to the A100 run) ----
        "--lora_rank", str(config.LORA_RANK),
        "--lora_alpha", str(config.LORA_ALPHA),
        "--lora_dropout", str(config.LORA_DROPOUT),
        "--target_modules", config.TARGET_MODULES,
        "--freeze_vit", config.FREEZE_VIT,
        # ---- schedule ----
        "--num_train_epochs", str(config.NUM_EPOCHS),
        "--learning_rate", config.LEARNING_RATE,
        "--weight_decay", config.WEIGHT_DECAY,
        "--warmup_ratio", config.WARMUP_RATIO,
        "--lr_scheduler_type", config.LR_SCHEDULER,
        "--per_device_train_batch_size", str(config.PER_DEVICE_BATCH_SIZE),
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(config.GRAD_ACCUM_STEPS),
        "--dataloader_num_workers", "4",
        "--max_length", str(config.MAX_LENGTH),
        "--gradient_checkpointing", config.GRAD_CHECKPOINTING,
        "--eval_strategy", "steps",
        "--eval_steps", str(config.EVAL_STEPS),
        "--save_strategy", "steps",
        "--save_steps", str(config.SAVE_STEPS),
        "--save_total_limit", str(config.SAVE_TOTAL_LIMIT),
        "--logging_steps", "5",
        "--seed", str(config.SEED),
        "--add_version", "false",
        "--output_dir", out_dir,
    ]
    if m["attn_impl"] is not None:
        cmd += ["--attn_impl", m["attn_impl"]]

    if rare_loss:
        # experimental: up-weight rare disease-code tokens via the external-plugin loss
        cmd += ["--external_plugins", os.path.join(config.HERE, "rare_loss.py"),
                "--loss_type", "rare_weighted"]

    if smoke:
        # Tiny end-to-end validation: 5 optimiser steps. max_steps (5) < eval/save_steps (100), so no
        # eval or checkpoint fires — it just proves the model loads, quantises and trains without OOM.
        cmd += ["--max_steps", "5"]
    else:
        last = find_last_checkpoint(out_dir)
        if last:
            print("Resuming from checkpoint:", last)
            cmd += ["--resume_from_checkpoint", last]
        else:
            print("Starting from scratch (no checkpoint in %s)." % out_dir)
    return cmd


def main():
    model_key = resolve_model_key(sys.argv)
    smoke = "--smoke" in sys.argv[2:]
    rare_loss = "--rare-loss" in sys.argv[2:]
    m = config.MODELS[model_key]

    os.chdir(config.DATA_ROOT)   # image paths in the jsonl are relative to DATA_ROOT
    assert os.path.exists(config.TRAIN_JSONL), \
        "Data not found. Did you run setup_gcp.py to unzip pan924_vlm.zip into %s ?" % config.DATA_ROOT

    env = with_max_pixels(make_env(), m["uses_max_pixels"])

    if rare_loss:
        # compute the rare-code token ids for THIS model's tokenizer and hand them to the plugin
        from rare_token_ids import compute_rare_token_ids
        ids, _ = compute_rare_token_ids(m["model"], config.RARE_CODES, config.COMMON_CODES)
        env["RARE_TOKEN_IDS"] = ",".join(map(str, ids))
        env["RARE_LOSS_WEIGHT"] = str(config.RARE_LOSS_WEIGHT)
        print("[rare-loss] up-weighting %d token ids x%.1f : %s" % (len(ids), config.RARE_LOSS_WEIGHT, ids))

    cmd = build_command(model_key, smoke, rare_loss)

    print("\n=== %s | %s | %s ===" % (model_key, m["model"], "SMOKE TEST (5 steps)" if smoke else "FULL TRAIN"))
    print("dataset:", config.train_dataset_path() if not smoke else config.TRAIN_JSONL,
          "| MAX_PIXELS:", config.MAX_PIXELS if m["uses_max_pixels"] else "n/a (fixed-res model)")
    print("output :", config.output_dir(model_key))
    print(">>> First minutes are SILENT (swift loads + quantises the model, preprocesses data) — NOT hung.")
    print(">>> Then loss prints every 5 steps. Watch VRAM in another shell: watch -n5 nvidia-smi\n")
    run_streaming(cmd, env=env)

    if smoke:
        print("\nSMOKE TEST PASSED — the config loads, quantises, and trains. Now run the full thing:")
        print("    python train.py %s" % model_key)
    else:
        print("\nDONE training %s. Next: python select_checkpoint.py %s" % (model_key, model_key))
    # CUDA OOM? Lower config.MAX_PIXELS (1003520 -> 802816 -> 602112) and re-run (it resumes).


if __name__ == "__main__":
    main()
