# PAN924 Code File I/O Guide / Hướng Dẫn Input-Output Từng File Code

Use this during the defense to explain each file in both English and Vietnamese.

Dùng file này khi bảo vệ để giải thích từng file code bằng cả tiếng Anh và tiếng Việt.

Main code root / Thư mục code chính:

- `/home/keytwelvelab/pan924/`

Main result root / Thư mục kết quả chính:

- `/home/keytwelvelab/pan924_runs/`

## Overall Pipeline / Luồng Tổng Thể

EN:
The project first builds a five-view VLM dataset from panoramic X-rays, converts it to model JSONL, optionally rebalances rare classes, trains VLMs with QLoRA, evaluates them, then serves the best model in a FastAPI demo.

VI:
Dự án đầu tiên tạo bộ dữ liệu VLM 5 góc nhìn từ ảnh X-quang toàn cảnh, chuyển sang JSONL cho từng model, có thể cân bằng các lớp hiếm, train VLM bằng QLoRA, đánh giá kết quả, rồi chạy demo FastAPI với model tốt nhất.

Key files / Các file chính:

- Dataset build / Tạo dataset: `vlm_report_dataset/scripts/build_5view_common_dataset.py`
- Dataset convert / Chuyển format dataset: `vlm_report_dataset/scripts/convert_dataset.py`
- Rare-class handling / Xử lý lớp hiếm: `gcp_l4/oversample.py`, `gcp_l4/augment_rare.py`, `gcp_l4/rare_loss.py`
- Training / Huấn luyện: `gcp_l4/train.py`, `gcp_l4/training/*/train.py`
- Evaluation / Đánh giá: `gcp_l4/evaluate.py`, `vlm_report_dataset/scripts/eval_report.py`
- Web demo / Demo web: `inference/app.py`, `inference/render_report.py`

## Dataset Pipeline / Pipeline Dữ Liệu

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/condition_maps.py`

Purpose / Mục đích:

- EN: Defines final dental condition classes and maps raw annotation labels to normalized labels.
- VI: Định nghĩa các lớp bệnh/trạng thái răng cuối cùng và gộp nhãn thô về nhãn chuẩn.

Input / Đầu vào:

- EN: Raw labels from annotation data, such as `Ri`, `RiM`, `TeM`, `I`.
- VI: Nhãn thô từ dữ liệu annotation, ví dụ `Ri`, `RiM`, `TeM`, `I`.

Output / Đầu ra:

- EN: Normalized condition codes such as `H`, `R`, `Te`, `M3i`, `Dc`, `P`.
- VI: Mã condition đã chuẩn hóa như `H`, `R`, `Te`, `M3i`, `Dc`, `P`.

Important code / Phần quan trọng:

- `RAW_TO_FINAL`
- `CONDITION_NAMES`
- `RARE_CONDITION_CLASSES`
- `normalize_condition`
- `condition_name`

Defense line / Câu nói khi bảo vệ:

- EN: This file proves how raw labels are merged into the final condition classes.
- VI: File này chứng minh phần gộp nhãn thô thành các lớp condition cuối cùng nằm ở đâu.

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/build_5view_common_dataset.py`

Purpose / Mục đích:

- EN: Builds the common five-view VLM dataset from annotated panoramic X-rays.
- VI: Tạo dataset VLM 5 góc nhìn từ ảnh X-quang toàn cảnh đã annotate.

Input / Đầu vào:

- EN: `annotations/dataset_final_v2.json`, original images in `images/`, and CLI options.
- VI: File annotation `annotations/dataset_final_v2.json`, ảnh gốc trong `images/`, và tham số dòng lệnh.

Output / Đầu ra:

- EN: Full image, four quadrant crops, `train.jsonl`, `val.jsonl`, `test.jsonl`, and metadata.
- VI: Ảnh full, 4 ảnh crop theo quadrant, các file `train.jsonl`, `val.jsonl`, `test.jsonl`, và metadata.

Important code / Phần quan trọng:

- `REGIONAL_PROMPT`, `FULL_PROMPT`: output schema for reports.
- `crop_boxes`: creates four region boxes.
- `tooth_payload`: converts one tooth annotation to FDI and condition.
- `teeth_in_box`: assigns teeth to each crop.
- `make_row`: creates one VLM training row.
- `main`: writes images and JSONL splits.

Defense line / Câu nói khi bảo vệ:

- EN: This is the file that creates the five views: one full panoramic image plus four regional crops.
- VI: Đây là file tạo 5 view: một ảnh toàn cảnh và bốn ảnh crop theo vùng.

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/generate_clinical_reports.py`

Purpose / Mục đích:

- EN: Generates report-style text from structured tooth findings.
- VI: Sinh câu báo cáo lâm sàng từ dữ liệu răng đã có cấu trúc.

Input / Đầu vào:

- EN: Tooth IDs, condition codes, and condition names.
- VI: FDI răng, mã condition, và tên condition.

Output / Đầu ra:

- EN: Regional comments and full-image summary text.
- VI: Comment cho từng vùng và phần summary cho ảnh full.

Important code / Phần quan trọng:

- EN: Functions that group abnormal teeth by condition and generate readable comments.
- VI: Các hàm gom răng bất thường theo condition và tạo câu mô tả dễ đọc.

Defense line / Câu nói khi bảo vệ:

- EN: This file explains how structured labels become natural report text.
- VI: File này giải thích cách nhãn có cấu trúc được chuyển thành câu báo cáo.

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/convert_dataset.py`

Purpose / Mục đích:

- EN: Converts the common dataset to model-specific JSONL formats.
- VI: Chuyển dataset chung sang format JSONL phù hợp với từng model.

Input / Đầu vào:

- EN: Common JSONL files from `vlm_report_dataset/common_balanced/`.
- VI: Các file JSONL chung trong `vlm_report_dataset/common_balanced/`.

Output / Đầu ra:

- EN: Converted files under `vlm_report_dataset/converted/<model>/`.
- VI: File đã convert trong `vlm_report_dataset/converted/<model>/`.

Important code / Phần quan trọng:

- `to_qwen`
- `to_llava`
- `convert_row`
- `main`

Defense line / Câu nói khi bảo vệ:

- EN: All models use the same common data, but this file converts it to each model's required format.
- VI: Tất cả model dùng cùng một dữ liệu gốc, file này chỉ chuyển sang format model yêu cầu.

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/rebalance_common.py`

Purpose / Mục đích:

- EN: Rebalances rare-condition samples in the common training split.
- VI: Cân bằng lại các sample chứa condition hiếm trong tập train.

Input / Đầu vào:

- EN: Common train JSONL and rare classes from `condition_maps.py`.
- VI: File train JSONL chung và danh sách rare class từ `condition_maps.py`.

Output / Đầu ra:

- EN: Rebalanced train JSONL and class-count summary.
- VI: File train đã cân bằng và thống kê số lượng từng lớp.

Important code / Phần quan trọng:

- EN: Rare-class counting, replication factor, replication cap, output writing.
- VI: Đếm lớp hiếm, tính hệ số nhân bản, giới hạn nhân bản, ghi file output.

Defense line / Câu nói khi bảo vệ:

- EN: Rare-class balancing is applied only to training data, not validation or test data.
- VI: Cân bằng lớp hiếm chỉ áp dụng cho train, không đụng vào validation/test.

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/validate_common_dataset.py`

Purpose / Mục đích:

- EN: Validates the common dataset before training.
- VI: Kiểm tra dataset chung trước khi train.

Input / Đầu vào:

- EN: JSONL rows, referenced image paths, and report JSON targets.
- VI: Các dòng JSONL, đường dẫn ảnh được tham chiếu, và JSON target của report.

Output / Đầu ra:

- EN: Terminal validation result or errors.
- VI: Kết quả kiểm tra trên terminal hoặc lỗi nếu dataset không hợp lệ.

Important code / Phần quan trọng:

- EN: Image path checks, JSON parsing, FDI validation, condition validation.
- VI: Kiểm tra đường dẫn ảnh, parse JSON, kiểm tra FDI, kiểm tra condition.

Defense line / Câu nói khi bảo vệ:

- EN: This is the dataset quality-control step.
- VI: Đây là bước kiểm tra chất lượng dataset trước khi train.

### `/home/keytwelvelab/pan924/vlm_report_dataset/scripts/eval_report.py`

Purpose / Mục đích:

- EN: Evaluates model predictions against gold dental reports.
- VI: Đánh giá prediction của model so với report ground truth.

Input / Đầu vào:

- EN: Gold JSONL and prediction JSONL.
- VI: File gold JSONL và file prediction JSONL.

Output / Đầu ra:

- EN: JSON validity, FDI F1, condition metrics, tooth accuracy, ROUGE-L, and per-class metrics.
- VI: Các metric như JSON validity, FDI F1, condition metrics, tooth accuracy, ROUGE-L, và metric từng lớp.

Important code / Phần quan trọng:

- `RARE = {"Dc", "Im", "P", "Rr", "M3f"}`
- JSON parsing and flattening logic.
- Metric computation.
- Per-condition report printing.

Defense line / Câu nói khi bảo vệ:

- EN: This is the main file used to compute the thesis evaluation metrics.
- VI: Đây là file chính để tính các metric đánh giá trong thesis.

## Dataset Training Tools / Công Cụ Hỗ Trợ Training Dataset

### `/home/keytwelvelab/pan924/vlm_report_dataset/training/tools/swift_pred_to_eval.py`

Purpose / Mục đích:

- EN: Converts raw ms-swift predictions into the format expected by `eval_report.py`.
- VI: Chuyển prediction thô từ ms-swift sang format mà `eval_report.py` đọc được.

Input / Đầu vào:

- EN: ms-swift prediction JSONL and gold/test JSONL.
- VI: File prediction JSONL từ ms-swift và file gold/test JSONL.

Output / Đầu ra:

- EN: Evaluation-ready JSONL.
- VI: File JSONL sẵn sàng cho bước evaluation.

Important code / Phần quan trọng:

- EN: Prediction extraction, row alignment, output writing.
- VI: Lấy text prediction, align với dòng gold, ghi file output.

Defense line / Câu nói khi bảo vệ:

- EN: This file connects model inference output to the evaluation script.
- VI: File này là cầu nối giữa output inference và script đánh giá.

### `/home/keytwelvelab/pan924/vlm_report_dataset/training/tools/compare_models.py`

Purpose / Mục đích:

- EN: Compares metrics across trained models.
- VI: So sánh metric giữa các model đã train.

Input / Đầu vào:

- EN: Metrics files under `/home/keytwelvelab/pan924_runs/<model>/`.
- VI: Các file metric trong `/home/keytwelvelab/pan924_runs/<model>/`.

Output / Đầu ra:

- EN: Overall comparison table and rare-condition recall table.
- VI: Bảng so sánh tổng thể và bảng recall cho condition hiếm.

Important code / Phần quan trọng:

- `RARE = ["Dc", "Im", "P", "Rr", "M3f"]`
- Metrics loading.
- Ranking/comparison table.

Defense line / Câu nói khi bảo vệ:

- EN: This supports the slide comparing the five VLMs.
- VI: File này hỗ trợ slide so sánh 5 model VLM.

## GCP L4 Training Pipeline / Pipeline Training Trên GCP L4

### `/home/keytwelvelab/pan924/gcp_l4/config.py`

Purpose / Mục đích:

- EN: Central configuration for training, evaluation, and inference.
- VI: File cấu hình trung tâm cho train, evaluate, và inference.

Input / Đầu vào:

- EN: Optional environment variables `PAN924_DATA_ROOT`, `PAN924_OUTPUT_ROOT`.
- VI: Biến môi trường tùy chọn `PAN924_DATA_ROOT`, `PAN924_OUTPUT_ROOT`.

Output / Đầu ra:

- EN: No direct file output; other scripts import it for settings.
- VI: Không tạo file trực tiếp; các script khác import file này để lấy cấu hình.

Important code / Phần quan trọng:

- `MODELS`
- `TRAIN_JSONL`, `VAL_JSONL`, `TEST_JSONL`
- QLoRA settings.
- LoRA settings.
- `RARE_CODES`, `COMMON_CODES`, `RARE_LOSS_WEIGHT`
- `train_dataset_path`

Defense line / Câu nói khi bảo vệ:

- EN: This ensures all models use the same shared experimental settings.
- VI: File này đảm bảo các model dùng cùng cấu hình thí nghiệm để so sánh công bằng.

### `/home/keytwelvelab/pan924/gcp_l4/common.py`

Purpose / Mục đích:

- EN: Shared helper functions for command execution and checkpoint handling.
- VI: Chứa hàm helper dùng chung cho chạy command và xử lý checkpoint.

Input / Đầu vào:

- EN: Model key, output path, environment variables.
- VI: Tên model, đường dẫn output, biến môi trường.

Output / Đầu ra:

- EN: Resolved model key, prepared environment, subprocess execution, checkpoint path.
- VI: Model key đã resolve, môi trường đã chuẩn bị, chạy subprocess, đường dẫn checkpoint.

Important code / Phần quan trọng:

- `resolve_model_key`
- `make_env`
- `with_max_pixels`
- `run_streaming`
- `find_last_checkpoint`

Defense line / Câu nói khi bảo vệ:

- EN: This file avoids duplicating common training utilities.
- VI: File này tránh lặp lại các tiện ích dùng chung trong nhiều script train/eval.

### `/home/keytwelvelab/pan924/gcp_l4/setup_gcp.py`

Purpose / Mục đích:

- EN: Prepares the server environment and dataset layout.
- VI: Chuẩn bị môi trường server và cấu trúc dataset.

Input / Đầu vào:

- EN: Dataset archive or existing dataset folder, Python environment.
- VI: File dataset nén hoặc folder dataset đã có, môi trường Python.

Output / Đầu ra:

- EN: Prepared dataset tree and optionally balanced training files.
- VI: Cây dataset đã chuẩn bị và có thể tạo thêm file train đã cân bằng.

Important code / Phần quan trọng:

- EN: Dataset extraction/checking and rare-class balancing call.
- VI: Giải nén/kiểm tra dataset và gọi bước cân bằng lớp hiếm.

Defense line / Câu nói khi bảo vệ:

- EN: This documents the setup process for the GCP L4 server.
- VI: File này mô tả cách setup server GCP L4 để train.

### `/home/keytwelvelab/pan924/gcp_l4/oversample.py`

Purpose / Mục đích:

- EN: Duplicates training rows that contain rare conditions.
- VI: Nhân bản các dòng train có chứa condition hiếm.

Input / Đầu vào:

- EN: `vlm_report_dataset/converted/qwen/train.jsonl` and settings from `config.py`.
- VI: File `vlm_report_dataset/converted/qwen/train.jsonl` và cấu hình từ `config.py`.

Output / Đầu ra:

- EN: `train_balanced.jsonl` and before/after class distribution.
- VI: File `train_balanced.jsonl` và thống kê phân bố lớp trước/sau.

Important code / Phần quan trọng:

- `flatten_conditions`
- Per-condition factor calculation.
- `REPLICATION_CAP`
- Output writing.

Defense line / Câu nói khi bảo vệ:

- EN: This is data-level rare-class balancing for the training split only.
- VI: Đây là cân bằng lớp hiếm ở mức dữ liệu, chỉ áp dụng cho tập train.

### `/home/keytwelvelab/pan924/gcp_l4/augment_rare.py`

Purpose / Mục đích:

- EN: Creates augmented image copies for rare-condition samples.
- VI: Tạo ảnh augment cho các sample chứa condition hiếm.

Input / Đầu vào:

- EN: Training JSONL, original images, rare-class settings.
- VI: File train JSONL, ảnh gốc, và cấu hình rare class.

Output / Đầu ra:

- EN: Augmented images and a balanced train JSONL.
- VI: Ảnh augment và file train JSONL đã cân bằng.

Important code / Phần quan trọng:

- EN: Image augmentation, replication logic, JSONL rewriting.
- VI: Hàm augment ảnh, logic nhân bản, ghi lại JSONL.

Defense line / Câu nói khi bảo vệ:

- EN: This is better than plain duplication because it gives the model varied images.
- VI: Cách này tốt hơn duplicate đơn thuần vì model thấy ảnh có biến đổi.

### `/home/keytwelvelab/pan924/gcp_l4/rare_token_ids.py`

Purpose / Mục đích:

- EN: Computes tokenizer IDs for rare condition codes.
- VI: Tính token ID tương ứng với các mã condition hiếm.

Input / Đầu vào:

- EN: Model tokenizer, `RARE_CODES`, `COMMON_CODES`.
- VI: Tokenizer của model, `RARE_CODES`, `COMMON_CODES`.

Output / Đầu ra:

- EN: Distinctive rare token IDs.
- VI: Danh sách token ID đặc trưng cho rare code.

Important code / Phần quan trọng:

- `CONTEXTS`
- `_ids_for`
- `compute_rare_token_ids`

Defense line / Câu nói khi bảo vệ:

- EN: This file converts rare labels into token IDs so the loss can target them.
- VI: File này chuyển nhãn hiếm thành token ID để loss có thể tăng trọng số đúng token.

### `/home/keytwelvelab/pan924/gcp_l4/rare_loss.py`

Purpose / Mục đích:

- EN: Implements rare-token weighted cross-entropy for ms-swift.
- VI: Cài đặt cross-entropy có trọng số cho token hiếm trong ms-swift.

Input / Đầu vào:

- EN: Model logits, labels, `RARE_TOKEN_IDS`, `RARE_LOSS_WEIGHT`.
- VI: Logits của model, labels, `RARE_TOKEN_IDS`, `RARE_LOSS_WEIGHT`.

Output / Đầu ra:

- EN: Custom loss value registered as `rare_weighted`.
- VI: Giá trị loss tùy chỉnh được đăng ký với tên `rare_weighted`.

Important code / Phần quan trọng:

- `_weighted_ce`
- `rare_weighted_loss`
- `_register`

Defense line / Câu nói khi bảo vệ:

- EN: This reduces model bias toward common labels such as healthy/restored.
- VI: File này giúp giảm bias của model về các nhãn phổ biến như healthy/restored.

### `/home/keytwelvelab/pan924/gcp_l4/train.py`

Purpose / Mục đích:

- EN: Generic command-line launcher for training one model with ms-swift.
- VI: File launcher chung để train một model bằng ms-swift.

Input / Đầu vào:

- EN: `python train.py <model>`, optional `--smoke`, optional `--rare-loss`.
- VI: Lệnh `python train.py <model>`, có thể thêm `--smoke`, `--rare-loss`.

Output / Đầu ra:

- EN: Checkpoints and logs in `/home/keytwelvelab/pan924_runs/<model>/`.
- VI: Checkpoint và log trong `/home/keytwelvelab/pan924_runs/<model>/`.

Important code / Phần quan trọng:

- `build_command`
- Rare-loss option.
- Resume checkpoint logic.
- `main`

Defense line / Câu nói khi bảo vệ:

- EN: This file builds the exact `swift sft` training command.
- VI: File này tạo command `swift sft` chính xác để train model.

### `/home/keytwelvelab/pan924/gcp_l4/run_model.py`

Purpose / Mục đích:

- EN: Runs the full training-to-evaluation pipeline for one model.
- VI: Chạy toàn bộ pipeline từ train đến đánh giá cho một model.

Input / Đầu vào:

- EN: Model key and optional train flags.
- VI: Tên model và các flag train tùy chọn.

Output / Đầu ra:

- EN: Training output, selected checkpoint, metrics, and plots.
- VI: Kết quả train, checkpoint được chọn, metric, và biểu đồ.

Important code / Phần quan trọng:

- EN: Sequential calls to `train.py`, `select_checkpoint.py`, `evaluate.py`, `visualize.py`.
- VI: Gọi lần lượt `train.py`, `select_checkpoint.py`, `evaluate.py`, `visualize.py`.

Defense line / Câu nói khi bảo vệ:

- EN: This file automates the full experiment for one model.
- VI: File này tự động hóa toàn bộ thí nghiệm cho một model.

### `/home/keytwelvelab/pan924/gcp_l4/run_model.sh`

Purpose / Mục đích:

- EN: Bash wrapper for the full one-model pipeline.
- VI: Shell script để chạy full pipeline cho một model.

Input / Đầu vào:

- EN: `bash run_model.sh <model> [train flags]`
- VI: Lệnh `bash run_model.sh <model> [train flags]`

Output / Đầu ra:

- EN: Same outputs as `run_model.py`.
- VI: Output giống `run_model.py`.

Important code / Phần quan trọng:

- EN: The order `train -> select checkpoint -> evaluate -> visualize`.
- VI: Thứ tự chạy `train -> chọn checkpoint -> evaluate -> visualize`.

Defense line / Câu nói khi bảo vệ:

- EN: This is only a convenience script; core logic is still in Python.
- VI: Đây chỉ là script tiện ích; logic chính vẫn nằm trong Python.

### `/home/keytwelvelab/pan924/gcp_l4/select_checkpoint.py`

Purpose / Mục đích:

- EN: Selects the best checkpoint after training.
- VI: Chọn checkpoint tốt nhất sau khi train.

Input / Đầu vào:

- EN: Checkpoint folders and trainer/eval state files.
- VI: Các thư mục checkpoint và file trạng thái train/eval.

Output / Đầu ra:

- EN: `best_checkpoint.txt`.
- VI: File `best_checkpoint.txt`.

Important code / Phần quan trọng:

- EN: Checkpoint scanning, eval-loss comparison, writing selected path.
- VI: Quét checkpoint, so sánh eval loss, ghi đường dẫn checkpoint tốt nhất.

Defense line / Câu nói khi bảo vệ:

- EN: Evaluation uses the best checkpoint, not a random saved checkpoint.
- VI: Bước đánh giá dùng checkpoint tốt nhất, không lấy ngẫu nhiên.

### `/home/keytwelvelab/pan924/gcp_l4/evaluate.py`

Purpose / Mục đích:

- EN: Runs inference and evaluation for a fine-tuned model.
- VI: Chạy inference và đánh giá cho model đã fine-tune.

Input / Đầu vào:

- EN: Model key, test JSONL, selected LoRA checkpoint.
- VI: Tên model, test JSONL, checkpoint LoRA đã chọn.

Output / Đầu ra:

- EN: Prediction JSONL, converted eval JSONL, metrics JSON, text report.
- VI: File prediction JSONL, eval JSONL đã convert, metrics JSON, report text.

Important code / Phần quan trọng:

- EN: Checkpoint loading, ms-swift inference, conversion, evaluation call.
- VI: Load checkpoint, chạy inference ms-swift, convert output, gọi script evaluate.

Defense line / Câu nói khi bảo vệ:

- EN: This is the main test-set evaluation pipeline for fine-tuned models.
- VI: Đây là pipeline đánh giá test set chính cho model đã fine-tune.

### `/home/keytwelvelab/pan924/gcp_l4/evaluate_base.py`

Purpose / Mục đích:

- EN: Evaluates a base model without LoRA fine-tuning.
- VI: Đánh giá model gốc chưa fine-tune LoRA.

Input / Đầu vào:

- EN: Model key and test JSONL.
- VI: Tên model và test JSONL.

Output / Đầu ra:

- EN: Base-model predictions and base metrics.
- VI: Prediction và metric của base model.

Important code / Phần quan trọng:

- EN: Inference without adapter and the same evaluation path as fine-tuned models.
- VI: Inference không dùng adapter và đánh giá bằng cùng một pipeline với model fine-tune.

Defense line / Câu nói khi bảo vệ:

- EN: This gives the zero-shot baseline.
- VI: File này tạo baseline zero-shot.

### `/home/keytwelvelab/pan924/gcp_l4/run_base_evals.py`

Purpose / Mục đích:

- EN: Runs base evaluation for multiple models.
- VI: Chạy đánh giá base cho nhiều model.

Input / Đầu vào:

- EN: List of model keys.
- VI: Danh sách tên model.

Output / Đầu ra:

- EN: Base evaluation outputs for each model.
- VI: Kết quả đánh giá base cho từng model.

Important code / Phần quan trọng:

- EN: Loop over model keys and call `evaluate_base.py`.
- VI: Loop qua các model và gọi `evaluate_base.py`.

Defense line / Câu nói khi bảo vệ:

- EN: This is the Python automation wrapper for base evaluation.
- VI: Đây là wrapper Python để tự động hóa đánh giá base model.

### `/home/keytwelvelab/pan924/gcp_l4/run_base_evals.sh`

Purpose / Mục đích:

- EN: Bash wrapper for base-model evaluation.
- VI: Shell script để chạy evaluation cho base model.

Input / Đầu vào:

- EN: `bash run_base_evals.sh internvl qwen`
- VI: Lệnh `bash run_base_evals.sh internvl qwen`

Output / Đầu ra:

- EN: Base metrics and `eval_base.log`.
- VI: Base metrics và file `eval_base.log`.

Important code / Phần quan trọng:

- EN: Sequential loop with `tee` logging.
- VI: Loop tuần tự và ghi log bằng `tee`.

Defense line / Câu nói khi bảo vệ:

- EN: It avoids running multiple GPU evaluations at the same time.
- VI: Script này tránh chạy nhiều evaluation cùng lúc trên một GPU.

### `/home/keytwelvelab/pan924/gcp_l4/visualize.py`

Purpose / Mục đích:

- EN: Creates plots from evaluation metrics.
- VI: Tạo biểu đồ từ kết quả evaluation.

Input / Đầu vào:

- EN: Metrics JSON and prediction/evaluation outputs.
- VI: File metrics JSON và output prediction/evaluation.

Output / Đầu ra:

- EN: Confusion matrix, per-condition plots, FDI recall charts, training curves.
- VI: Confusion matrix, biểu đồ từng condition, FDI recall chart, đường cong training.

Important code / Phần quan trọng:

- EN: Metrics loading, condition plotting, rare-class marking, image saving.
- VI: Load metric, vẽ theo condition, đánh dấu class hiếm, lưu ảnh.

Defense line / Câu nói khi bảo vệ:

- EN: This generates figures used for analysis and slides.
- VI: File này tạo hình dùng cho phân tích và slide.

## Model-Specific Training Files / File Train Riêng Từng Model

### `/home/keytwelvelab/pan924/gcp_l4/training/internvl3_8b/train.py`

Purpose / Mục đích:

- EN: Direct Python-API training script for InternVL3-8B.
- VI: File train trực tiếp bằng Python API cho InternVL3-8B.

Input / Đầu vào:

- EN: Converted train/val JSONL, InternVL3-8B base model, rare-token settings.
- VI: Train/val JSONL đã convert, base model InternVL3-8B, cấu hình rare-token.

Output / Đầu ra:

- EN: LoRA checkpoints under `/home/keytwelvelab/pan924_runs/internvl/`.
- VI: LoRA checkpoint trong `/home/keytwelvelab/pan924_runs/internvl/`.

Important code / Phần quan trọng:

- `MODEL`
- `OUTPUT_DIR`
- `configure_environment`
- `build_training_arguments`

Defense line / Câu nói khi bảo vệ:

- EN: This is the standalone training file for the best model.
- VI: Đây là file train độc lập cho model tốt nhất.

### `/home/keytwelvelab/pan924/gcp_l4/training/qwen2_5_vl_7b/train.py`

Purpose / Mục đích:

- EN: Direct training script for Qwen2.5-VL-7B.
- VI: File train trực tiếp cho Qwen2.5-VL-7B.

Input / Đầu vào:

- EN: Converted train/val JSONL and Qwen2.5-VL-7B base model.
- VI: Train/val JSONL đã convert và base model Qwen2.5-VL-7B.

Output / Đầu ra:

- EN: LoRA checkpoints under `/home/keytwelvelab/pan924_runs/qwen/`.
- VI: LoRA checkpoint trong `/home/keytwelvelab/pan924_runs/qwen/`.

Important code / Phần quan trọng:

- EN: Model ID, output folder, QLoRA/LoRA settings, rare loss settings.
- VI: Model ID, output folder, cấu hình QLoRA/LoRA, cấu hình rare loss.

Defense line / Câu nói khi bảo vệ:

- EN: This is the 7B Qwen training setup used for comparison.
- VI: Đây là cấu hình train Qwen 7B dùng để so sánh.

### `/home/keytwelvelab/pan924/gcp_l4/training/qwen2_5_vl_3b/train.py`

Purpose / Mục đích:

- EN: Direct training script for Qwen2.5-VL-3B.
- VI: File train trực tiếp cho Qwen2.5-VL-3B.

Input / Đầu vào:

- EN: Converted train/val JSONL and Qwen2.5-VL-3B base model.
- VI: Train/val JSONL đã convert và base model Qwen2.5-VL-3B.

Output / Đầu ra:

- EN: LoRA checkpoints under `/home/keytwelvelab/pan924_runs/qwen3b/`.
- VI: LoRA checkpoint trong `/home/keytwelvelab/pan924_runs/qwen3b/`.

Important code / Phần quan trọng:

- EN: Model ID, output folder, QLoRA/LoRA settings, rare loss settings.
- VI: Model ID, output folder, cấu hình QLoRA/LoRA, cấu hình rare loss.

Defense line / Câu nói khi bảo vệ:

- EN: This is the smaller Qwen model used for size/performance comparison.
- VI: Đây là model Qwen nhỏ hơn dùng để so sánh kích thước và hiệu năng.

### `/home/keytwelvelab/pan924/gcp_l4/training/phi3_5_vision/train.py`

Purpose / Mục đích:

- EN: Direct training script for Phi-3.5-Vision.
- VI: File train trực tiếp cho Phi-3.5-Vision.

Input / Đầu vào:

- EN: Converted train/val JSONL and Phi-3.5-Vision base model.
- VI: Train/val JSONL đã convert và base model Phi-3.5-Vision.

Output / Đầu ra:

- EN: LoRA checkpoints under `/home/keytwelvelab/pan924_runs/phi/`.
- VI: LoRA checkpoint trong `/home/keytwelvelab/pan924_runs/phi/`.

Important code / Phần quan trọng:

- EN: Model ID, attention implementation, QLoRA/LoRA settings, rare loss settings.
- VI: Model ID, attention implementation, cấu hình QLoRA/LoRA, cấu hình rare loss.

Defense line / Câu nói khi bảo vệ:

- EN: This file shows how Phi-3.5-Vision was trained under the same task setting.
- VI: File này cho thấy Phi-3.5-Vision được train trong cùng setting thí nghiệm.

### `/home/keytwelvelab/pan924/gcp_l4/training/paligemma2_3b/train.py`

Purpose / Mục đích:

- EN: Direct training script for PaliGemma2-3B.
- VI: File train trực tiếp cho PaliGemma2-3B.

Input / Đầu vào:

- EN: Converted train/val JSONL and PaliGemma2-3B base model.
- VI: Train/val JSONL đã convert và base model PaliGemma2-3B.

Output / Đầu ra:

- EN: LoRA checkpoints under `/home/keytwelvelab/pan924_runs/paligemma/`.
- VI: LoRA checkpoint trong `/home/keytwelvelab/pan924_runs/paligemma/`.

Important code / Phần quan trọng:

- EN: Model ID, fixed-resolution handling, QLoRA/LoRA settings, rare loss settings.
- VI: Model ID, xử lý fixed resolution, cấu hình QLoRA/LoRA, cấu hình rare loss.

Defense line / Câu nói khi bảo vệ:

- EN: This is the PaliGemma training setup used in the model comparison.
- VI: Đây là cấu hình train PaliGemma dùng trong phần so sánh model.

## Web Inference Demo / Demo Web Inference

### `/home/keytwelvelab/pan924/inference/app.py`

Purpose / Mục đích:

- EN: FastAPI app for uploading a panoramic X-ray and generating a dental report.
- VI: Ứng dụng FastAPI để upload ảnh X-quang toàn cảnh và sinh report nha khoa.

Input / Đầu vào:

- EN: Image upload at `/analyze` and fine-tuned InternVL3 adapter.
- VI: Ảnh upload vào endpoint `/analyze` và adapter InternVL3 đã fine-tune.

Output / Đầu ra:

- EN: HTML page showing image, four regions, tooth findings, summary, and raw JSON.
- VI: Trang HTML hiển thị ảnh, 4 vùng, finding từng răng, summary, và JSON thô.

Important code / Phần quan trọng:

- `MODEL`
- `ADAPTER`
- `FULL_REPORT_PROMPT`
- `get_engine`
- `run_model`
- `/analyze` route

Defense line / Câu nói khi bảo vệ:

- EN: This is the live demo interface for the fine-tuned model.
- VI: Đây là giao diện demo trực tiếp cho model đã fine-tune.

### `/home/keytwelvelab/pan924/inference/render_report.py`

Purpose / Mục đích:

- EN: Parses, repairs, validates, and renders the model's JSON report.
- VI: Parse, sửa lỗi nhẹ, validate, và hỗ trợ render JSON report của model.

Input / Đầu vào:

- EN: Raw model output string.
- VI: Chuỗi output thô từ model.

Output / Đầu ra:

- EN: Parsed report dictionary or parse error.
- VI: Dictionary report đã parse hoặc lỗi parse.

Important code / Phần quan trọng:

- `CONDITION_NAMES`
- `_is_full_report_shape`
- `_loads_json_with_repairs`
- `_parse_full_prediction_loose`
- `parse_full_prediction`
- `parse_error`

Defense line / Câu nói khi bảo vệ:

- EN: This file makes the demo robust to small JSON mistakes while still requiring four regions and summary.
- VI: File này giúp demo chịu được lỗi JSON nhỏ nhưng vẫn bắt buộc có đủ 4 vùng và summary.

## Runtime Patch / Patch Runtime

### `/home/keytwelvelab/pan924/patches/sitecustomize.py`

Purpose / Mục đích:

- EN: Compatibility patch for the installed Transformers version.
- VI: Patch tương thích cho version Transformers đang cài trên server.

Input / Đầu vào:

- EN: Python import process and `transformers.cache_utils.DynamicCache`.
- VI: Quá trình import Python và `transformers.cache_utils.DynamicCache`.

Output / Đầu ra:

- EN: Adds missing runtime attributes/methods if needed.
- VI: Thêm thuộc tính/phương thức bị thiếu khi cần.

Important code / Phần quan trọng:

- `DynamicCache.seen_tokens`
- `DynamicCache.get_max_length`

Defense line / Câu nói khi bảo vệ:

- EN: This is not model logic; it is only an environment compatibility patch.
- VI: Đây không phải logic model; chỉ là patch để môi trường chạy ổn định.

