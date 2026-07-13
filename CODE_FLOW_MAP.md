# Code Flow Map for Thesis Slides

This file maps the defense-slide pipeline to concrete files in this repository.
Use it as the quick answer sheet when asked "where is that in code?"

## Dataset Source and Cleaning

- PAN924/InReDD source schema and project context: `README.md`
- Raw label -> final 12-class condition merge rules:
  - `code/dataset_pipeline/condition_maps.py`
  - `code/faster_rcnn_disease/scripts/prepare_disease_dataset.py`
- Merge rules shown in the slides:
  - `Ri`, `RiM`, `TeM` -> `Te`
  - `I` -> `M3i`
  - final classes are `H, R, Te, CpuM, M3i, M3f, Di, C, Rr, P, Im, Dc`

## Five-View VLM Dataset

- Create full image + four quadrant crops:
  - `code/dataset_pipeline/build_5view_common_dataset.py`
- Region names:
  - `image_upper_left`
  - `image_upper_right`
  - `image_lower_left`
  - `image_lower_right`
- Common JSONL row format and prompt/target creation:
  - `code/dataset_pipeline/build_5view_common_dataset.py`
- Template comments and full-image summary generation:
  - `code/dataset_pipeline/generate_clinical_reports.py`
- Common dataset sanity checks:
  - `code/dataset_pipeline/validate_common_dataset.py`

## Class Imbalance Handling

- Train-only rare-condition oversampling:
  - `code/dataset_pipeline/rebalance_common.py`
  - `code/gcp_l4/oversample.py`
  - `code/gcp_l4/augment_rare.py`
- Rare-token loss:
  - `code/gcp_l4/rare_loss.py`
  - `code/gcp_l4/rare_token_ids.py`
  - `code/gcp_l4/config.py`
- Rare condition set used in evaluation and comparison:
  - `code/dataset_pipeline/eval_report.py`
  - `code/dataset_pipeline/compare_models.py`

## Model Format Conversion

- Convert common rows into model JSONL formats:
  - `code/dataset_pipeline/convert_dataset.py`
- Swift prediction adapter:
  - `code/dataset_pipeline/swift_pred_to_eval.py`

## VLM Training and Testing

- Shared L4 training launcher/config:
  - `code/gcp_l4/train.py`
  - `code/gcp_l4/config.py`
  - `code/gcp_l4/run_model.py`
- Explicit per-model QLoRA scripts:
  - `code/gcp_l4/training/qwen2_5_vl_7b/train.py`
  - `code/gcp_l4/training/qwen2_5_vl_3b/train.py`
  - `code/gcp_l4/training/internvl3_8b/train.py`
  - `code/gcp_l4/training/phi3_5_vision/train.py`
  - `code/gcp_l4/training/paligemma2_3b/train.py`
- Checkpoint selection:
  - `code/gcp_l4/select_checkpoint.py`
- Test-set evaluation:
  - `code/gcp_l4/evaluate.py`
  - `code/dataset_pipeline/eval_report.py`
- Model comparison table:
  - `code/dataset_pipeline/compare_models.py`

## Faster R-CNN Baselines

- FDI/tooth detector dataset preparation:
  - `code/faster_rcnn_fdi/scripts/prepare_frcnn_dataset.py`
- Disease/condition detector dataset preparation:
  - `code/faster_rcnn_disease/scripts/prepare_disease_dataset.py`
- Detector training:
  - `code/faster_rcnn_fdi/scripts/train_frcnn.py`
  - `code/faster_rcnn_disease/scripts/train_frcnn.py`
- Detector inference:
  - `code/faster_rcnn_fdi/scripts/infer_frcnn.py`
  - `code/faster_rcnn_disease/scripts/infer_frcnn.py`
- Benchmark sweeps and plots:
  - `code/faster_rcnn_fdi/scripts/benchmark_frcnn_models.py`
  - `code/faster_rcnn_disease/scripts/benchmark_frcnn_models.py`
  - `code/faster_rcnn_disease/scripts/plot_benchmark.py`

## Final Web Inference Demo

- FastAPI upload page and InternVL3 inference:
  - `code/inference/app.py`
- Report renderer and robust JSON repair:
  - `code/inference/render_report.py`
- Web-app requirements and run instructions:
  - `code/inference/README.md`

## Results Referenced by Slides

- Per-model metrics and figures:
  - `results/qwen/`
  - `results/qwen3b/`
  - `results/internvl/`
  - `results/phi/`
  - `results/paligemma/`
- Thesis figures and LaTeX source:
  - `report/`
