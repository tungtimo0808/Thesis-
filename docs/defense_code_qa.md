# Defense Code Q&A Map

This document maps likely thesis-defense questions to the exact code locations in the repository.

## Quick Index

Repository root: `Thesis-repo/`

Main summary file: `CODE_FLOW_MAP.md`

## Q&A

### Q1. Where is the overall slide-to-code map?

Answer: Use `CODE_FLOW_MAP.md`.

Key files:
- `CODE_FLOW_MAP.md`

What to say: "This file maps each slide pipeline block to the exact source files: dataset preparation, five views, class merging, training, evaluation, Faster R-CNN baselines, and web inference."

### Q2. Where is the code for merging raw condition classes into the final 12 classes?

Answer: The merge rules are in `code/dataset_pipeline/condition_maps.py` and are also applied in the Faster R-CNN disease-preparation script.

Key files:
- `code/dataset_pipeline/condition_maps.py`
- `code/faster_rcnn_disease/scripts/prepare_disease_dataset.py`

Important rules:
- `Ri -> Te`
- `RiM -> Te`
- `TeM -> Te`
- `I -> M3i`

What to say: "The final condition set is fixed in `CONDITION_NAMES` / `FINAL_CONDITION_CLASSES`, and `normalize_condition()` applies the raw-label remap."

### Q3. Where is the code for the final 12 condition classes?

Answer: The authoritative list is in `code/dataset_pipeline/condition_maps.py`.

Key files:
- `code/dataset_pipeline/condition_maps.py`
- `code/faster_rcnn_disease/scripts/prepare_disease_dataset.py`

Final classes:
- `H, R, Te, CpuM, M3i, M3f, Di, C, Rr, P, Im, Dc`

### Q4. Where is the code that creates the 5 views?

Answer: The five-view dataset builder is `code/dataset_pipeline/build_5view_common_dataset.py`.

Key files:
- `code/dataset_pipeline/build_5view_common_dataset.py`

What it creates:
- `full.jpg`
- `image_upper_left.jpg`
- `image_upper_right.jpg`
- `image_lower_left.jpg`
- `image_lower_right.jpg`

What to say: "This script crops each panoramic image into four overlapping quadrants and also keeps the full panoramic image, giving 5 views per image."

### Q5. Where are the four quadrant names defined?

Answer: They are defined in the dataset builder and reused by the inference/reporting code.

Key files:
- `code/dataset_pipeline/build_5view_common_dataset.py`
- `code/inference/render_report.py`
- `code/inference/app.py`

Region names:
- `image_upper_left`
- `image_upper_right`
- `image_lower_left`
- `image_lower_right`

### Q6. Where is the code that creates the JSON report target?

Answer: The JSON target schemas are generated in the five-view dataset builder and evaluated in the report evaluator.

Key files:
- `code/dataset_pipeline/build_5view_common_dataset.py`
- `code/dataset_pipeline/eval_report.py`
- `code/inference/render_report.py`

Schemas:
- Regional report: `{"region": "...", "teeth": [...], "comment": "..."}`
- Full report: `{"regions": {... four regions ...}, "summary": "..."}`

### Q7. Where is the code that fills comments and summary text?

Answer: Template-based report text is generated in `generate_clinical_reports.py`; the web app also renders the final report.

Key files:
- `code/dataset_pipeline/generate_clinical_reports.py`
- `code/inference/render_report.py`
- `code/inference/app.py`

### Q8. Where is the train/validation/test split created?

Answer: For the VLM report dataset, split-by-image is in the five-view builder. For Faster R-CNN baselines, deterministic split logic is in the prepare scripts.

Key files:
- `code/dataset_pipeline/build_5view_common_dataset.py`
- `code/faster_rcnn_fdi/scripts/prepare_frcnn_dataset.py`
- `code/faster_rcnn_disease/scripts/prepare_disease_dataset.py`

What to say: "The VLM pipeline splits images before generating the 5 views, so crops from the same panoramic image do not leak across train/val/test."

### Q9. Where is the code for converting the common dataset to model formats?

Answer: Use `code/dataset_pipeline/convert_dataset.py`.

Key files:
- `code/dataset_pipeline/convert_dataset.py`
- `code/dataset_pipeline/swift_pred_to_eval.py`

Model families:
- Qwen2.5-VL
- InternVL3
- Phi-3.5-vision
- PaliGemma 2
- LLaVA-style format support

### Q10. Where is the code for rare-condition oversampling?

Answer: There are two implementations: the compact dataset-pipeline script and the GCP training utility.

Key files:
- `code/dataset_pipeline/rebalance_common.py`
- `code/gcp_l4/oversample.py`
- `code/gcp_l4/augment_rare.py`
- `code/gcp_l4/config.py`

What to say: "Only the training split is rebalanced; validation and test stay unchanged."

### Q11. Where is the rare-token weighted loss implemented?

Answer: It is implemented as an ms-swift external loss plugin.

Key files:
- `code/gcp_l4/rare_loss.py`
- `code/gcp_l4/rare_token_ids.py`
- `code/gcp_l4/config.py`
- Per-model scripts under `code/gcp_l4/training/*/train.py`

What to say: "The loss up-weights tokens corresponding to rare condition codes, without changing validation/test data."

### Q12. Where is the QLoRA training setup?

Answer: The shared launcher/config is under `code/gcp_l4/`, and each model also has a self-contained training script.

Key files:
- `code/gcp_l4/config.py`
- `code/gcp_l4/train.py`
- `code/gcp_l4/run_model.py`
- `code/gcp_l4/training/qwen2_5_vl_7b/train.py`
- `code/gcp_l4/training/qwen2_5_vl_3b/train.py`
- `code/gcp_l4/training/internvl3_8b/train.py`
- `code/gcp_l4/training/phi3_5_vision/train.py`
- `code/gcp_l4/training/paligemma2_3b/train.py`

What to say: "All model scripts use the same controlled QLoRA recipe: 4-bit base model, LoRA adapters, same split, and same evaluation process."

### Q13. Where is model evaluation done?

Answer: Inference and metric scoring are handled by the GCP evaluation script and the dataset-pipeline evaluator.

Key files:
- `code/gcp_l4/evaluate.py`
- `code/dataset_pipeline/eval_report.py`
- `code/dataset_pipeline/swift_pred_to_eval.py`
- `code/dataset_pipeline/compare_models.py`
- `code/gcp_l4/visualize.py`

Metrics:
- JSON validity
- FDI detection F1
- condition accuracy on detected teeth
- micro/macro/weighted F1
- tooth accuracy
- text ROUGE-L

### Q14. Where are the final model-comparison results?

Answer: Final metrics and figures are tracked under `results/`.

Key folders:
- `results/internvl/`
- `results/paligemma/`
- `results/qwen/`
- `results/qwen3b/`
- `results/phi/`

### Q15. Where is the Faster R-CNN FDI baseline?

Answer: The FDI detector code is under `code/faster_rcnn_fdi/`.

Key files:
- `code/faster_rcnn_fdi/scripts/prepare_frcnn_dataset.py`
- `code/faster_rcnn_fdi/scripts/train_frcnn.py`
- `code/faster_rcnn_fdi/scripts/infer_frcnn.py`
- `code/faster_rcnn_fdi/scripts/benchmark_frcnn_models.py`
- `code/faster_rcnn_fdi/configs/train_config.yaml`

### Q16. Where is the Faster R-CNN disease/condition baseline?

Answer: The condition detector code is under `code/faster_rcnn_disease/`.

Key files:
- `code/faster_rcnn_disease/scripts/prepare_disease_dataset.py`
- `code/faster_rcnn_disease/scripts/train_frcnn.py`
- `code/faster_rcnn_disease/scripts/infer_frcnn.py`
- `code/faster_rcnn_disease/scripts/benchmark_frcnn_models.py`
- `code/faster_rcnn_disease/scripts/plot_benchmark.py`
- `code/faster_rcnn_disease/configs/train_config.yaml`

### Q17. Where is the web demo / final inference app?

Answer: The FastAPI app is under `code/inference/`.

Key files:
- `code/inference/app.py`
- `code/inference/render_report.py`
- `code/inference/README.md`
- `code/inference/requirements.txt`

What to say: "`app.py` loads InternVL3 with the fine-tuned adapter, accepts an uploaded panoramic X-ray, asks for the full four-region report, and renders the result."

### Q18. Where is the robust JSON parsing for model output?

Answer: It is in `code/inference/render_report.py`.

Key files:
- `code/inference/render_report.py`

What to say: "The parser accepts valid JSON, repairs common missing-comma model errors, and has a loose fallback for schema-like output so the demo can still render a full report."

### Q19. Where is the thesis report source and figures?

Answer: The written thesis and report figures are in `report/`.

Key files/folders:
- `report/report.tex`
- `report/references.bib`
- `report/*.png`
- `report/USTH.jpg`

### Q20. What file should I open first before the defense?

Answer: Open `CODE_FLOW_MAP.md` first, then this document.

Key files:
- `CODE_FLOW_MAP.md`
- `docs/defense_code_qa.md`
- `docs/defense_code_qa.docx`

What to say: "The codebase is organized so each slide pipeline step maps to a concrete source file, and the Q&A file lists the exact paths."
