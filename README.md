# Thesis — Vision-Language Models for Automatic Dental Report Generation

Bachelor thesis (USTH). The project fine-tunes and compares several Vision-Language
Models (VLMs) for generating structured dental reports from panoramic X-ray images,
and compares them against a Faster R-CNN recognition baseline.

## Repository layout

```
report/      LaTeX thesis (report.tex + references.bib) and figures
code/        Source code
  gcp_l4/              VLM training/eval pipeline (QLoRA on a single L4 GPU, ms-swift)
  faster_rcnn_fdi/     Faster R-CNN for tooth (FDI) detection
  faster_rcnn_disease/ Faster R-CNN for condition (disease) detection
  dataset_pipeline/    Dataset conversion and evaluation scripts
results/     Per-model test metrics, text reports, and figures (no model weights)
  qwen/ qwen3b/ internvl/ phi/ paligemma/
```

## Models compared

| Key | Model | Size |
|---|---|---|
| qwen | Qwen2.5-VL-7B-Instruct | ~8B |
| qwen3b | Qwen2.5-VL-3B-Instruct | ~3B |
| internvl | InternVL3-8B | ~8B |
| phi | Phi-3.5-vision-instruct | ~4.2B |
| paligemma | PaliGemma 2 (3B, 448px) | ~3B |

All models use the same data, the same split, and the same QLoRA recipe, so the
comparison is controlled. The best model on the test set is InternVL3-8B.

## Not included

Model checkpoints, the image dataset, the `.jsonl` data files, and the Python
environment are intentionally left out (size and licensing). The dataset is the
PAN924 panoramic dental X-ray dataset.
