"""Make rare diseases easier to learn: oversample training reports that contain rare conditions.

THE PROBLEM (measured on this dataset)
  H has 37,775 tooth-occurrences in train; Dc has 804, Rr 996, Im 968 ... (H:Dc ~ 47:1). The
  language-model loss is dominated by the common H/R teeth, so the model is biased to relabel rare
  teeth as H/R. On the qwen test run this showed up as F1 = 0.00 for Dc, Rr, C even though
  condition_acc_on_detected was 0.71 (i.e. classification works WHEN a tooth is detected — the
  rare classes just rarely "win").

THE FIX (data-level, the most reliable lever)
  Duplicate whole reports so that each condition reaches ~OVERSAMPLE_TARGET tooth-occurrences. A
  report's replication factor = the max factor required by ANY condition it contains (so a report
  holding the rarest disease is duplicated the most), capped at REPLICATION_CAP to avoid overfitting
  a handful of reports. Reports with only common conditions stay at 1x.

  This does NOT touch val/test, so evaluation is still on the true distribution — only the training
  signal is rebalanced. Run it once; train.py then uses train_balanced.jsonl automatically.

Usage:
  python oversample.py              # writes vlm_report_dataset/converted/qwen/train_balanced.jsonl
"""
import json
import math
import os
from collections import Counter

import config


def flatten_conditions(report):
    """Return the list of condition codes present in a gold report (handles both schemas)."""
    out = []
    if not isinstance(report, dict):
        return out
    if "regions" in report and isinstance(report["regions"], dict):
        for payload in report["regions"].values():
            for t in (payload or {}).get("teeth", []) if isinstance(payload, dict) else []:
                if isinstance(t, dict) and t.get("condition"):
                    out.append(t["condition"])
    elif "teeth" in report:
        for t in report.get("teeth", []):
            if isinstance(t, dict) and t.get("condition"):
                out.append(t["condition"])
    return out


def gold_text(o):
    for m in o.get("messages", []):
        if m.get("role") == "assistant":
            c = m["content"]
            return c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c else None)
    return None


def main():
    src = os.path.join(config.DATA_ROOT, config.TRAIN_JSONL)
    dst = os.path.join(config.DATA_ROOT, config.TRAIN_BALANCED_JSONL)
    assert os.path.exists(src), "train.jsonl not found at " + src

    rows = []
    with open(src, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                rows.append(json.loads(ln))

    # 1) Count tooth-occurrences per condition across the whole train split.
    conds_per_row = []
    total = Counter()
    for o in rows:
        try:
            conds = flatten_conditions(json.loads(gold_text(o)))
        except Exception:
            conds = []
        conds_per_row.append(conds)
        total.update(conds)

    # 2) Per-condition oversample factor needed to reach the target (>=1).
    factor = {c: max(1, math.ceil(config.OVERSAMPLE_TARGET / n)) for c, n in total.items()}

    # 3) Each report's replication = max factor among its conditions, capped.
    out_rows = []
    rep_hist = Counter()
    for o, conds in zip(rows, conds_per_row):
        rep = 1 if not conds else min(config.REPLICATION_CAP, max(factor[c] for c in set(conds)))
        rep_hist[rep] += 1
        out_rows.extend([o] * rep)

    with open(dst, "w", encoding="utf-8") as f:
        for o in out_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # 4) Report the before/after distribution so the rebalancing is auditable (not a black box).
    after = Counter()
    for o, conds in zip(rows, conds_per_row):
        rep = 1 if not conds else min(config.REPLICATION_CAP, max(factor[c] for c in set(conds)))
        after.update({c: v * rep for c, v in Counter(conds).items()})

    print("Reports: %d -> %d  (replication histogram: %s)" %
          (len(rows), len(out_rows), dict(sorted(rep_hist.items()))))
    print("%-8s %10s %10s %8s" % ("cond", "teeth_before", "teeth_after", "xgain"))
    for c, n in total.most_common():
        print("%-8s %10d %10d %7.1fx" % (c, n, after[c], after[c] / n))
    print("\nWrote", dst)
    print("train.py will use it automatically (config.USE_BALANCED_TRAIN = True).")


if __name__ == "__main__":
    main()
