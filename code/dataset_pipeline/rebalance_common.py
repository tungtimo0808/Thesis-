"""Oversample rare-condition regional reports in TRAIN only."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from condition_maps import RARE_CONDITION_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance common train JSONL by rare conditions")
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common"))
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_report_dataset/common_balanced"))
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--cap", type=int, default=8)
    parser.add_argument("--seed", type=int, default=924)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def target_report(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][-1]["content"])


def conditions(report: dict[str, Any]) -> list[str]:
    out = []
    if isinstance(report.get("regions"), dict):
        payloads = report["regions"].values()
    else:
        payloads = [report]
    for payload in payloads:
        for tooth in payload.get("teeth", []):
            if tooth.get("condition"):
                out.append(tooth["condition"])
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    train = read_jsonl(args.input_dir / "train.jsonl")
    counts = Counter()
    row_conds = []
    for row in train:
        conds = conditions(target_report(row))
        row_conds.append(conds)
        counts.update(conds)
    factors = {c: min(args.cap, max(1, round(args.target / max(1, counts[c])))) for c in RARE_CONDITION_CLASSES}
    balanced = []
    for row, conds in zip(train, row_conds):
        factor = max([factors.get(c, 1) for c in conds] or [1])
        balanced.extend([row] * factor)
    random.shuffle(balanced)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", balanced)
    for split in ("val", "test"):
        write_jsonl(args.output_dir / ("%s.jsonl" % split), read_jsonl(args.input_dir / ("%s.jsonl" % split)))
    summary = {
        "seed": args.seed,
        "rare_classes": sorted(RARE_CONDITION_CLASSES),
        "original_train_rows": len(train),
        "balanced_train_rows": len(balanced),
        "condition_counts": dict(counts),
        "factors": factors,
        "note": "Only train is oversampled; val/test are copied unchanged.",
    }
    meta = args.output_dir / "metadata" / "rebalance_summary.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
