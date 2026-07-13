# PAN924 VLM training on a GCP L4 (24 GB) — 4-bit QLoRA

Train the 5 VLMs on a Google Cloud **L4** VM over SSH (VS Code Remote). No Colab, no Google Drive —
everything is on the VM's local disk. Checkpoints are saved locally and training **auto-resumes**, so
you can run it under `tmux` and the job survives SSH disconnects / closing your laptop.

| key | model |
|---|---|
| `qwen` | Qwen/Qwen2.5-VL-7B-Instruct |
| `internvl` | OpenGVLab/InternVL3-8B |
| `llava` | llava-hf/llava-onevision-qwen2-7b-ov-hf |
| `phi` | microsoft/Phi-3.5-vision-instruct (attn = eager) |
| `paligemma` | google/paligemma2-3b-pt-448 (gated — `huggingface-cli login` first) |

## Why 4-bit QLoRA here (and why it's still correct)
A 7–8B model in bf16 needs ~28–40 GB and will **not** fit the L4's 24 GB. So this path uses **4-bit
QLoRA** (NF4 + double-quant, bf16 compute): weights drop to ~5 GB, leaving room for activations and a
**high image resolution**. Quality loss vs bf16 LoRA is small (~1–2 pts), and — crucially — **all 5
models use the same 4-bit config**, so they stay comparable to each other. Everything else (LoRA
rank/alpha, lr, effective batch 16, eval/save every 100 steps) is identical to the A100 notebooks.

> ⚠️ The A100 qwen run used **bf16** at higher fidelity. A 4-bit-L4 number is **not** directly
> comparable to that bf16-A100 number. For a fair table, run **all 5** here on the L4 (including qwen).

## 0. Upload to the VM
Put these three files anywhere (e.g. your home `~`):
- `pan924_vlm.zip` — the dataset (~2.3 GB: jsonl + 4620 images + eval scripts)
- `pan924_gcp_l4.zip` — this code
- `setup_gcp.py` — the setup script (it's also inside the zip, but you need a copy to bootstrap)

## 1. Setup (unzip + install + verify GPU + build balanced data)
```bash
python3 setup_gcp.py
```
It extracts both zips into `~/pan924/`, pip-installs the stack, prints torch/GPU/bitsandbytes/swift
versions, and builds the rare-class-balanced training file. Result layout:
```
~/pan924/vlm_report_dataset/...   (data)
~/pan924/gcp_l4/*.py              (code)
~/pan924_runs/<model>/            (checkpoints + results, created while training)
```

## 2. ALWAYS smoke-test a model first (~2 min, ~free)
```bash
cd ~/pan924/gcp_l4
python train.py qwen --smoke
```
This runs **5 optimiser steps** end-to-end. If it loads, quantises and trains without OOM, the config
is sound — *then* commit to the full run. This is the guard against burning hours/money on a broken
config.

## 3. Full pipeline (inside tmux so it survives disconnects)
```bash
tmux new -s pan924
cd ~/pan924/gcp_l4
bash run_model.sh qwen        # train -> select checkpoint -> evaluate -> visualize
```
Detach with `Ctrl-b d`; reattach later with `tmux attach -t pan924`. Re-running after an interruption
**resumes** training from the last checkpoint (eval/viz are cheap to re-run).

Run the steps individually if you prefer:
```bash
python train.py qwen              # full train (auto-resume)
python select_checkpoint.py qwen  # pick lowest-val-loss checkpoint (instant, free)
python evaluate.py qwen           # test metrics -> metrics_qwen_test.json + report_qwen_test.txt
python visualize.py qwen          # cm_counts / cm_norm / per_condition PNGs
```
Repeat for `internvl`, `llava`, `phi`, `paligemma`. Then compare:
```bash
cd ~/pan924
python vlm_report_dataset/training/tools/compare_models.py ~/pan924_runs/*/metrics_*_test.json
```

## Outputs per model (`~/pan924_runs/<model>/`)
- `checkpoint-*/` — LoRA adapters (resume + the chosen one for eval)
- `best_checkpoint.txt` — the selected checkpoint
- `metrics_<model>_test.json` — full metrics (for compare_models.py)
- `report_<model>_test.txt` — human-readable report (overall + per-condition)
- `cm_counts_<model>.png`, `cm_norm_<model>.png`, `per_condition_<model>.png`

## Rare-class handling (your question: "make rare diseases easier to recognise")
The dataset is very imbalanced (H : Dc ≈ 39 : 1). Several levers are built in:

1. **Image augmentation** (`augment_rare.py`, used by setup by default) — instead of duplicating a
   rare report's pixels (overfit), it keeps the original + makes augmented image copies (brightness/
   contrast/sharpness/±4° rotation; **never a flip** — that would swap left/right and break FDI). The
   rare disease is seen under varied exposure → generalises better. Writes `train_balanced.jsonl`.
   - Plain duplication fallback: `oversample.py` (no Pillow needed).
   - **Honest limit:** every report still carries H/R teeth, so the class ratio only improves ~39:1 →
     ~27:1. Augmentation makes those extra rare samples *useful* rather than memorised, but it is not
     a magic fix.

2. **Rare-token loss up-weighting** (`rare_loss.py`, **opt-in**: `train.py <model> --rare-loss`) —
   multiplies the cross-entropy on rare disease-code tokens by `RARE_LOSS_WEIGHT` (default 3.0). This
   attacks the bias directly (the model relabelling rare teeth as H/R) **without** dragging H/R along
   the way oversampling does. Implemented as an ms-swift external-plugin loss. The math is unit-tested
   (`python rare_loss.py`), but the swift plugin API can vary by version — **validate with
   `python train.py qwen --rare-loss --smoke` first** (it surfaces any API mismatch in ~2 min).

3. **High resolution** (`MAX_PIXELS = 1003520`) — small lesions (caries `Dc`, residual root `Rr`) need
   detail; 4-bit weights free the VRAM to afford it.

4. **Detection first** — the qwen diagnostic showed `condition_acc_on_detected = 0.71` (the model
   classifies fine *once it finds a tooth*); rare F1 was ~0 mostly because teeth were missed / JSON was
   truncated. `INFER_MAX_NEW_TOKENS = 1280` (vs the buggy 768) fixes truncation, lifting recall for
   **all** classes including rare ones.

**Recommended combo:** augmentation (on) + `--rare-loss`. Run it as:
```bash
python train.py qwen --rare-loss --smoke   # validate config + plugin (~2 min)
python train.py qwen --rare-loss           # full run
```
To push harder: raise `OVERSAMPLE_TARGET`/`REPLICATION_CAP` (re-run `augment_rare.py`),
`RARE_LOSS_WEIGHT`, or `NUM_EPOCHS` (2→3) in `config.py`, then retrain.

> ⚠️ `run_model.sh` does NOT pass `--rare-loss` by default (it's experimental). Either run the steps
> by hand with the flag, or add it to the `python train.py "$MODEL"` line in `run_model.sh`.

## Tuning knobs (`config.py`)
- **OOM during training** → lower `MAX_PIXELS` (1003520 → 802816 → 602112), re-run (resumes). OOM
  shows up at step 0, so it costs ~nothing to find the right value.
- **OOM during eval** → lower `INFER_MAX_BATCH_SIZE` (8 → 4).
- **train ~2× faster** if VRAM is comfortable → `PER_DEVICE_BATCH_SIZE = 2`, `GRAD_ACCUM_STEPS = 8`
  (still effective 16).

## Troubleshooting
- **`torch.cuda.is_available()` is False but `nvidia-smi` works** — torch's CUDA build doesn't match
  the driver. Reinstall torch for your CUDA, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
- **bitsandbytes import / CUDA error** — `pip install -U bitsandbytes`; it must match the torch CUDA.
- **swift downloads from `modelscope.cn` (slow)** — `USE_HF=1` wasn't set; the scripts set it, but if
  you call swift by hand, `export USE_HF=1` first.
- **gated paligemma** — accept the license on Hugging Face, then `huggingface-cli login` on the VM.
- **vLLM (optional ~10× faster eval)** — this VM is CUDA 13.x, so a matching vLLM build *could* work
  here (it failed on Colab's CUDA 12). It's optional; `pt` is the reliable default. Only try it if eval
  time is a real problem.
