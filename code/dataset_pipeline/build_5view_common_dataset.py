"""Build the common five-view PAN924 VLM dataset.

Input:
  annotations/dataset_final_v2.json
  images/<panoramic>.jpg

Output:
  vlm_report_dataset/common/images/<sample>/<full|quadrant>.jpg
  vlm_report_dataset/common/{train,val,test}.jsonl
  vlm_report_dataset/common/metadata/build_summary.json

The five views match the slide flow: one full panoramic image plus four
overlapping image-space quadrants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from condition_maps import condition_name, normalize_condition

REGIONS = ("image_upper_left", "image_upper_right", "image_lower_left", "image_lower_right")

REGIONAL_PROMPT = (
    "<image>\n"
    "This is the {region} crop of a panoramic dental radiograph.\n"
    "Identify the visible annotated teeth using FDI notation and report each tooth condition.\n"
    "Return only valid JSON with this schema:\n"
    '{"region":"string","teeth":[{"fdi":"string","condition":"string",'
    '"condition_name":"string"}],"comment":"string"}'
)

FULL_PROMPT = (
    "<image>\n"
    "This is a full panoramic dental radiograph.\n"
    "Divide the image into four image-space regions: image_upper_left, image_upper_right, "
    "image_lower_left, and image_lower_right.\n"
    "For each region, list the visible annotated teeth using FDI notation and report each "
    "tooth condition.\n"
    "Return only valid JSON with this schema:\n"
    '{"regions":{"image_upper_left":{"teeth":[{"fdi":"string","condition":"string",'
    '"condition_name":"string"}],"comment":"string"},"image_upper_right":{"teeth":[{"fdi":'
    '"string","condition":"string","condition_name":"string"}],"comment":"string"},'
    '"image_lower_left":{"teeth":[{"fdi":"string","condition":"string","condition_name":'
    '"string"}],"comment":"string"},"image_lower_right":{"teeth":[{"fdi":"string",'
    '"condition":"string","condition_name":"string"}],"comment":"string"}},"summary":"string"}'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build common 5-view VLM report dataset")
    parser.add_argument("--root", type=Path, default=Path("."), help="PAN924 project root")
    parser.add_argument("--input-json", type=Path, default=Path("annotations/dataset_final_v2.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=924)
    parser.add_argument("--overlap-x", type=float, default=0.10)
    parser.add_argument("--overlap-y", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def stable_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha1(("%s:%s" % (seed, key)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def split_name(file_name: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    stem = Path(file_name).stem
    bucket = stable_bucket(stem, seed)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def crop_boxes(width: int, height: int, overlap_x: float, overlap_y: float) -> dict[str, tuple[int, int, int, int]]:
    x_mid = width // 2
    y_mid = height // 2
    dx = int(width * overlap_x / 2)
    dy = int(height * overlap_y / 2)
    return {
        "image_upper_left": (0, 0, min(width, x_mid + dx), min(height, y_mid + dy)),
        "image_upper_right": (max(0, x_mid - dx), 0, width, min(height, y_mid + dy)),
        "image_lower_left": (0, max(0, y_mid - dy), min(width, x_mid + dx), height),
        "image_lower_right": (max(0, x_mid - dx), max(0, y_mid - dy), width, height),
    }


def tooth_center(tooth: dict[str, Any]) -> tuple[float, float] | None:
    bbox = tooth.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    return x + w / 2.0, y + h / 2.0


def tooth_payload(tooth: dict[str, Any]) -> dict[str, str] | None:
    fdi = tooth.get("fdi_label")
    cond = normalize_condition(tooth.get("disease_label"))
    if fdi is None or cond is None:
        return None
    return {"fdi": str(fdi), "condition": cond, "condition_name": condition_name(cond)}


def teeth_in_box(teeth: list[dict[str, Any]], box: tuple[int, int, int, int]) -> list[dict[str, str]]:
    x1, y1, x2, y2 = box
    out = []
    seen = set()
    for tooth in teeth:
        center = tooth_center(tooth)
        payload = tooth_payload(tooth)
        if center is None or payload is None:
            continue
        cx, cy = center
        if x1 <= cx <= x2 and y1 <= cy <= y2 and payload["fdi"] not in seen:
            seen.add(payload["fdi"])
            out.append(payload)
    return sorted(out, key=lambda t: int(t["fdi"]))


def region_comment(teeth: list[dict[str, str]]) -> str:
    if not teeth:
        return "No annotated teeth are visible in this region."
    abnormal = [t for t in teeth if t["condition"] != "H"]
    if not abnormal:
        return "The annotated teeth in this region are healthy."
    grouped: dict[str, list[str]] = {}
    for tooth in abnormal:
        grouped.setdefault(tooth["condition_name"], []).append(tooth["fdi"])
    parts = ["%s on teeth %s" % (name, ", ".join(fdis)) for name, fdis in grouped.items()]
    return "This region shows " + "; ".join(parts) + ". The remaining annotated teeth are healthy."


def full_summary(regions: dict[str, dict[str, Any]]) -> str:
    abnormal = []
    for payload in regions.values():
        for tooth in payload["teeth"]:
            if tooth["condition"] != "H":
                abnormal.append(tooth)
    if not abnormal:
        return "Overall, the annotated teeth are healthy."
    grouped: dict[str, list[str]] = {}
    for tooth in abnormal:
        grouped.setdefault(tooth["condition_name"], []).append(tooth["fdi"])
    parts = ["%s on teeth %s" % (name, ", ".join(fdis)) for name, fdis in grouped.items()]
    return "Overall, the annotated findings include " + "; ".join(parts) + "."


def make_row(sample_id: str, image_rel: str, source_image: str, view: str, task: str, prompt: str, target: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": sample_id,
        "image": image_rel,
        "source_image": source_image,
        "view": view,
        "task": task,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ],
        "metadata": metadata,
    }


def main() -> None:
    args = parse_args()
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1")

    root = args.root
    data = json.loads((root / args.input_json).read_text(encoding="utf-8"))
    out_dir = root / args.output_dir
    image_out_dir = out_dir / "images"
    metadata_dir = out_dir / "metadata"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = {"train": [], "val": [], "test": []}
    image_counts = {"train": 0, "val": 0, "test": 0}

    for item in data.get("images", []):
        file_name = str(item.get("file_name", "")).strip()
        if not file_name:
            continue
        source = root / args.image_dir / file_name
        if not source.exists():
            continue
        split = split_name(file_name, args.train_ratio, args.val_ratio, args.seed)
        image_counts[split] += 1
        sample_stem = Path(file_name).stem
        sample_dir = image_out_dir / sample_stem
        sample_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(source) as img:
            img = img.convert("RGB")
            width, height = img.size
            boxes = crop_boxes(width, height, args.overlap_x, args.overlap_y)

            full_rel = str(args.output_dir / "images" / sample_stem / "full.jpg")
            img.save(root / full_rel, quality=args.jpeg_quality)

            region_payloads = {}
            for region, box in boxes.items():
                crop_rel = str(args.output_dir / "images" / sample_stem / (region + ".jpg"))
                img.crop(box).save(root / crop_rel, quality=args.jpeg_quality)
                region_teeth = teeth_in_box(item.get("teeth", []), box)
                target = {"region": region, "teeth": region_teeth, "comment": region_comment(region_teeth)}
                region_payloads[region] = {"teeth": region_teeth, "comment": target["comment"]}
                rows[split].append(make_row(
                    "%s_%s_report" % (sample_stem, region),
                    crop_rel,
                    file_name,
                    region,
                    "regional_report",
                    REGIONAL_PROMPT.replace("{region}", region),
                    target,
                    {"width": width, "height": height, "crop_boxes": {k: list(v) for k, v in boxes.items()}},
                ))

            full_target = {"regions": region_payloads, "summary": full_summary(region_payloads)}
            rows[split].append(make_row(
                "%s_full_report" % sample_stem,
                full_rel,
                file_name,
                "full",
                "full_quadrant_report",
                FULL_PROMPT,
                full_target,
                {"width": width, "height": height, "crop_boxes": {k: list(v) for k, v in boxes.items()}},
            ))

    for split, split_rows in rows.items():
        with (out_dir / ("%s.jsonl" % split)).open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "seed": args.seed,
        "overlap_x": args.overlap_x,
        "overlap_y": args.overlap_y,
        "image_counts": image_counts,
        "sample_counts": {k: len(v) for k, v in rows.items()},
        "views_per_image": 5,
        "regions": list(REGIONS),
    }
    (metadata_dir / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
