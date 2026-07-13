"""Convert common PAN924 report rows to model-specific JSONL formats.

The ms-swift training path uses the Qwen-style format for all five model families;
the other formats are kept here as reproducibility evidence for the conversion step
shown in the slides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert common report dataset")
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common_balanced"))
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_report_dataset/converted"))
    parser.add_argument("--models", nargs="+", default=["qwen", "internvl", "phi", "paligemma", "llava"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_qwen(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "messages": row["messages"],
        "images": [row["image"]],
        "metadata": row.get("metadata", {}),
    }


def to_llava(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "image": row["image"],
        "conversations": [
            {"from": "human", "value": row["messages"][0]["content"]},
            {"from": "gpt", "value": row["messages"][1]["content"]},
        ],
        "metadata": row.get("metadata", {}),
    }


def convert_row(row: dict[str, Any], model: str) -> dict[str, Any]:
    if model in {"qwen", "internvl", "phi", "paligemma"}:
        return to_qwen(row)
    if model == "llava":
        return to_llava(row)
    raise ValueError("unknown model format: %s" % model)


def main() -> None:
    args = parse_args()
    for model in args.models:
        for split in ("train", "val", "test"):
            rows = [convert_row(row, model) for row in read_jsonl(args.input_dir / ("%s.jsonl" % split))]
            write_jsonl(args.output_dir / model / ("%s.jsonl" % split), rows)
            print(model, split, len(rows))


if __name__ == "__main__":
    main()
