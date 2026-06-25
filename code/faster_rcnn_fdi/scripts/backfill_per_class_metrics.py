"""Backfill per-class and confusion metrics for saved FDI Faster R-CNN runs.

This is an evaluate-only script. It loads each run's `best.pth`, evaluates the
test split, and augments the existing `test_results.json` with:
- test_macro_f1
- test_weighted_f1
- test_per_class
- confusion_matrix
- confusion_labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from frcnn_common import build_faster_rcnn_model, load_yaml, save_json, set_seed
from frcnn_dataset import CocoDetectionDataset, collate_fn
from train_frcnn import evaluate_per_class_and_confusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill FDI per-class metrics from checkpoints")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("faster_rcnn_fdi/configs/train_config.yaml"),
        help="Base config path",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("faster_rcnn_fdi/outputs/benchmark/benchmark_results.json"),
        help="Benchmark summary containing run metadata",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional run names to process")
    parser.add_argument("--force", action="store_true", help="Recompute even when per-class metrics exist")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_run_overrides(config: dict[str, Any], row: dict[str, Any]) -> None:
    model_cfg = config.setdefault("model", {})
    data_cfg = config.setdefault("data", {})
    train_cfg = config.setdefault("train", {})
    project_cfg = config.setdefault("project", {})

    model_cfg["architecture"] = str(row["architecture"])
    model_cfg["trainable_backbone_layers"] = int(row["trainable_backbone_layers"])
    model_cfg["pretrained"] = False
    data_cfg["min_size"] = int(row.get("min_size", data_cfg.get("min_size", 512)))
    data_cfg["max_size"] = int(row.get("max_size", data_cfg.get("max_size", 768)))
    project_cfg["output_dir"] = str(row["output_dir"])
    train_cfg["device"] = "cpu"


def update_results_summary(path: Path, final_results: dict[str, Any]) -> None:
    if not path.exists():
        return
    payload = load_json(path)
    if isinstance(payload.get("final"), dict):
        payload["final"].update(
            {
                "test_macro_f1": final_results["test_macro_f1"],
                "test_weighted_f1": final_results["test_weighted_f1"],
                "test_per_class": final_results["test_per_class"],
                "confusion_matrix": final_results["confusion_matrix"],
                "confusion_labels": final_results["confusion_labels"],
            }
        )
        save_json(path, payload)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = (root / args.config).resolve()
    benchmark_path = (root / args.benchmark).resolve()
    benchmark = load_json(benchmark_path)

    base_config = load_yaml(config_path)
    set_seed(int(base_config.get("project", {}).get("seed", 42)))

    data_cfg = base_config.get("data", {})
    test_ann = (root / data_cfg["test_annotations"]).resolve()
    images_root = (root / data_cfg["images_root"]).resolve()
    test_ds = CocoDetectionDataset(images_root=images_root, annotation_path=test_ann)
    id_to_name = {int(c["id"]): str(c["name"]) for c in test_ds.categories}
    num_fg_classes = len(test_ds.categories)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)

    eval_cfg = base_config.get("eval", {})
    score_threshold = float(eval_cfg.get("score_threshold", base_config.get("inference", {}).get("score_threshold", 0.3)))
    iou_threshold = float(eval_cfg.get("iou_threshold", 0.5))
    test_max_batches = int(eval_cfg.get("test_max_batches", 0))

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    wanted_runs = set(args.runs or [])

    for row in benchmark.get("runs", []):
        run_name = str(row["run_name"])
        if wanted_runs and run_name not in wanted_runs:
            continue

        output_dir = (root / str(row["output_dir"])).resolve()
        test_results_path = output_dir / "test_results.json"
        best_ckpt = output_dir / "best.pth"
        if not test_results_path.exists() or not best_ckpt.exists():
            print(f"[skip] {run_name}: missing test_results.json or best.pth")
            continue

        final_results = load_json(test_results_path)
        if final_results.get("test_per_class") and not args.force:
            print(f"[skip] {run_name}: per-class metrics already exist")
            continue

        run_config = json.loads(json.dumps(base_config))
        apply_run_overrides(run_config, row)
        model = build_faster_rcnn_model(run_config).to(device)
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])

        print(f"[eval] {run_name} on {device.type}")
        per_class = evaluate_per_class_and_confusion(
            model=model,
            loader=test_loader,
            device=device,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            max_batches=test_max_batches,
            num_classes=num_fg_classes,
            id_to_name=id_to_name,
        )

        final_results.update(
            {
                "test_macro_f1": per_class["macro_f1"],
                "test_weighted_f1": per_class["weighted_f1"],
                "test_per_class": per_class["per_class"],
                "confusion_matrix": per_class["confusion_matrix"],
                "confusion_labels": per_class["confusion_labels"],
            }
        )
        save_json(test_results_path, final_results)
        update_results_summary(output_dir / "results_summary.json", final_results)
        print(f"[ok] {run_name}: macro_f1={per_class['macro_f1']} weighted_f1={per_class['weighted_f1']}")


if __name__ == "__main__":
    main()
