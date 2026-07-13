"""Sanity-check common PAN924 report JSONL files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from condition_maps import CONDITION_NAMES

REGION_KEYS = {"image_upper_left", "image_upper_right", "image_lower_left", "image_lower_right"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate common report dataset")
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--out", type=Path, default=Path("vlm_report_dataset/common/metadata/validation_report.json"))
    return parser.parse_args()


def validate_target(target: dict) -> list[str]:
    errors = []
    targets = []
    if isinstance(target.get("regions"), dict):
        missing = REGION_KEYS - set(target["regions"])
        if missing:
            errors.append("missing regions: " + ",".join(sorted(missing)))
        for region, payload in target["regions"].items():
            for tooth in payload.get("teeth", []):
                targets.append((region, tooth))
    elif isinstance(target.get("teeth"), list):
        for tooth in target["teeth"]:
            targets.append((target.get("region", ""), tooth))
    else:
        errors.append("unknown target schema")
    for _, tooth in targets:
        if not tooth.get("fdi"):
            errors.append("missing fdi")
        if tooth.get("condition") not in CONDITION_NAMES:
            errors.append("unknown condition: %s" % tooth.get("condition"))
    return errors


def main() -> None:
    args = parse_args()
    summary = {"splits": {}, "errors": []}
    counts = Counter()
    for split in ("train", "val", "test"):
        path = args.input_dir / ("%s.jsonl" % split)
        rows = 0
        if not path.exists():
            summary["errors"].append("%s missing" % path)
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows += 1
            row = json.loads(line)
            image = row.get("image")
            if image and not Path(image).exists():
                summary["errors"].append("%s:%d missing image %s" % (path, line_no, image))
            target = json.loads(row["messages"][-1]["content"])
            for err in validate_target(target):
                summary["errors"].append("%s:%d %s" % (path, line_no, err))
            counts[row.get("task", "unknown")] += 1
        summary["splits"][split] = rows
    summary["task_counts"] = dict(counts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
