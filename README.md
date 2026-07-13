# PAN924 Server Code

This repository mirrors the code structure used for the live server demo in:

`/home/keytwelvelab/pan924`

It contains code and documentation only. Images, JSONL datasets, checkpoints, and run artifacts are kept on the server and are intentionally not pushed to GitHub.

Main folders:

- `gcp_l4/` - training, evaluation, checkpoint selection, rare-class handling, and visualization.
- `gcp_l4/training/` - model-specific QLoRA training scripts.
- `vlm_report_dataset/scripts/` - dataset building, label merging, conversion, validation, rebalancing, and evaluation.
- `vlm_report_dataset/training/tools/` - prediction conversion and model comparison tools.
- `inference/` - FastAPI web demo.
- `patches/` - runtime compatibility patch.

Useful defense documents:

- `code_file_io_guide_vi_en.md`
- `code_file_io_guide_vi_en.docx`

