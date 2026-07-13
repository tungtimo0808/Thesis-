"""Regenerate template comments and summaries for common-format JSONL rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill comment/summary fields in common JSONL")
    parser.add_argument("--input-dir", type=Path, default=Path("vlm_report_dataset/common/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("vlm_report_dataset/common"))
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def comment_for(teeth: list[dict[str, str]]) -> str:
    if not teeth:
        return "No annotated teeth are visible in this region."
    abnormal = [t for t in teeth if t.get("condition") != "H"]
    if not abnormal:
        return "The annotated teeth in this region are healthy."
    groups: dict[str, list[str]] = {}
    for tooth in abnormal:
        groups.setdefault(tooth.get("condition_name") or tooth.get("condition", "unknown"), []).append(str(tooth.get("fdi")))
    return "This region shows " + "; ".join("%s on teeth %s" % (k, ", ".join(v)) for k, v in groups.items()) + "."


def fill_target(target: dict[str, Any]) -> dict[str, Any]:
    if isinstance(target.get("regions"), dict):
        for payload in target["regions"].values():
            payload["comment"] = comment_for(payload.get("teeth", []))
        findings = []
        for payload in target["regions"].values():
            findings.extend([t for t in payload.get("teeth", []) if t.get("condition") != "H"])
        target["summary"] = "Overall, the annotated teeth are healthy." if not findings else "Overall, the annotated findings include " + ", ".join("%s %s" % (t.get("fdi"), t.get("condition_name") or t.get("condition")) for t in findings) + "."
    elif isinstance(target.get("teeth"), list):
        target["comment"] = comment_for(target["teeth"])
    return target


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        out_rows = []
        for row in read_rows(args.input_dir / ("%s.jsonl" % split)):
            target = json.loads(row["messages"][-1]["content"])
            row["messages"][-1]["content"] = json.dumps(fill_target(target), ensure_ascii=False)
            out_rows.append(row)
        with (args.output_dir / ("%s.jsonl" % split)).open("w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(split, len(out_rows))


if __name__ == "__main__":
    main()
