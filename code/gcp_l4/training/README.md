# Training code (one file per model)

Each model has its own self-contained training script. To find the code that trained a
given model, open the matching folder. Every script writes out all hyperparameters in
full (no hidden shared config) and trains with QLoRA on a single NVIDIA L4 (24 GB) GPU
using the ms-swift Python API (`from swift.llm import sft_main, TrainArguments`).

| Model | Folder | Script | Rare-token loss | Attention | Image size |
|---|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | `qwen2_5_vl_7b/` | `train.py` | yes (weight 3.0) | default | native (MAX_PIXELS) |
| Qwen2.5-VL-3B-Instruct | `qwen2_5_vl_3b/` | `train.py` | yes (weight 3.0) | default | native (MAX_PIXELS) |
| InternVL3-8B | `internvl3_8b/` | `train.py` | yes (weight 3.0) | default | 448px tiles (MAX_PIXELS) |
| Phi-3.5-vision-instruct | `phi3_5_vision/` | `train.py` | yes (weight 3.0) | eager | dynamic crop (MAX_PIXELS) |
| PaliGemma 2 (3B, 448px) | `paligemma2_3b/` | `train.py` | yes (weight 3.0) | default | fixed 448px |

## Shared settings (identical across all five scripts)

- Fine-tuning: QLoRA — frozen base in 4-bit NF4 (double quant, bfloat16 compute),
  trainable bfloat16 LoRA adapters.
- LoRA: rank 8, alpha 32, dropout 0.1, on all linear layers; vision encoder frozen.
- Schedule: 2 epochs, learning rate 1e-4, cosine scheduler, warmup ratio 0.05,
  weight decay 0.1.
- Batch: per-device batch size 1, gradient accumulation 16 (effective batch size 16),
  max sequence length 4096, gradient checkpointing on.
- Data: balanced PAN924 split (`vlm_report_dataset/converted/qwen/train.jsonl`,
  5,817 rows); validation on `val.jsonl`.
- Checkpoints: evaluate and save every 100 steps, keep the last 12; seed 924.

## Per-model differences (the only things that change)

- **Model id** and **output folder**.
- **Rare-token loss**: applied for all five models (the loss on the tokens that spell a
  rare disease code is multiplied by 3.0, via the `rare_loss.py` plugin in the parent
  folder).
- **Attention**: Phi-3.5-vision uses `attn_impl="eager"` for stability.
- **Image size**: four models keep a native/variable size capped by `MAX_PIXELS=401408`;
  PaliGemma 2 uses a fixed 448px input and does not read `MAX_PIXELS`.

## How to run

The image paths inside the `.jsonl` files are relative to the dataset root, so run from
the folder that contains `vlm_report_dataset/`:

```bash
cd ~/pan924
python gcp_l4/training/internvl3_8b/train.py     # example: train InternVL3-8B
```

PaliGemma 2 is gated: accept its license on Hugging Face and run `hf auth login` first.

## Relation to the original pipeline

These per-model scripts make the training of each model explicit and easy to locate.
They use the same ms-swift pipeline and the same hyperparameters as the original generic
launcher (`../train.py` + `../config.py`, driven by `../run_model.py`), with one script
per model so the configuration of each model can be read on its own.
