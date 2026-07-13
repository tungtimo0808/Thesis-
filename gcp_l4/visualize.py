"""Confusion matrix + per-condition charts, built from the SAME predictions evaluate.py scored"""
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")             # headless VM: render to file, no display needed
import matplotlib.pyplot as plt

import config
from common import resolve_model_key


def load_eval_helpers():
    """Import flatten/parse from the dataset's own eval script so parsing matches the metrics exactly."""
    sys.path.insert(0, os.path.join(config.DATA_ROOT, "vlm_report_dataset", "scripts"))
    from eval_report import flatten, parse_json_maybe, gold_target_string, RARE
    return flatten, parse_json_maybe, gold_target_string, RARE


def main():
    model_key = resolve_model_key(sys.argv)
    out_dir = config.output_dir(model_key)
    os.chdir(config.DATA_ROOT)

    marker = os.path.join(out_dir, "best_checkpoint.txt")
    if not os.path.exists(marker):
        sys.exit("Run select_checkpoint.py + evaluate.py first.")
    ckpt = open(marker, encoding="utf-8").read().strip()
    pred_out = os.path.join(ckpt, "preds_test.jsonl")
    metrics_out = os.path.join(out_dir, "metrics_%s_test.json" % model_key)
    for p in (pred_out, metrics_out):
        if not os.path.exists(p):
            sys.exit("Missing %s — run evaluate.py %s first." % (p, model_key))

    flatten, parse_json_maybe, gold_target_string, RARE = load_eval_helpers()

    # 1) gold (test) + predictions -> {(region, fdi): condition}
    gold = {}
    with open(os.path.join(config.DATA_ROOT, config.TEST_JSONL), encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                o = json.loads(ln)
                g = parse_json_maybe(gold_target_string(o))
                if g is not None:
                    gold[o["id"]] = flatten(g)
    preds = {}
    with open(pred_out, encoding="utf-8") as f:
        for ln in f:
            if ln.strip():
                o = json.loads(ln)
                p = parse_json_maybe(o.get("prediction"))
                preds[o["id"]] = flatten(p) if p is not None else {}

    # 2) one row per tooth; missed/hallucinated teeth get the '(none)' class
    NONE = "(none)"
    y_true, y_pred = [], []
    for sid, g in gold.items():
        p = preds.get(sid, {})
        for key in set(g) | set(p):
            y_true.append(g.get(key, NONE))
            y_pred.append(p.get(key, NONE))

    support = {}
    for t in y_true:
        support[t] = support.get(t, 0) + 1
    labels = sorted({l for l in set(y_true) | set(y_pred) if l != NONE},
                    key=lambda c: -support.get(c, 0)) + [NONE]
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    def plot_cm(title, fname, normalize=False):
        M = cm.astype(float)
        if normalize:
            rs = M.sum(axis=1, keepdims=True)
            M = np.divide(M, rs, out=np.zeros_like(M), where=rs != 0)
        side = max(6, len(labels) * 0.75)
        fig, ax = plt.subplots(figsize=(side, side))
        im = ax.imshow(M, cmap="Blues")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Gold (true)"); ax.set_title(title)
        thr = M.max() / 2 if M.max() else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                if M[i, j] > 0:
                    txt = ("%.2f" % M[i, j]) if normalize else ("%d" % cm[i, j])
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > thr else "black")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = os.path.join(out_dir, fname)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        print("saved", out)

    plot_cm("%s - confusion matrix (counts)" % model_key, "cm_counts_%s.png" % model_key)
    plot_cm("%s - confusion matrix (row-normalized = recall per gold class)" % model_key,
            "cm_norm_%s.png" % model_key, normalize=True)

    # 3) per-condition P/R/F1 bars
    metrics = json.load(open(metrics_out, encoding="utf-8"))
    pc = metrics["per_condition"]
    conds = sorted(pc, key=lambda c: -pc[c]["support"])
    P = [pc[c]["P"] for c in conds]; R = [pc[c]["R"] for c in conds]; F = [pc[c]["F1"] for c in conds]
    x = np.arange(len(conds)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(conds) * 0.85), 4.8))
    ax.bar(x - w, P, w, label="Precision"); ax.bar(x, R, w, label="Recall"); ax.bar(x + w, F, w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels([c + (" *" if c in RARE else "") for c in conds], rotation=45, ha="right")
    ax.set_ylim(0, 1.1); ax.set_ylabel("score"); ax.legend(loc="lower right")
    ax.set_title("%s - per-condition P/R/F1 on test  (* = rare class)" % model_key)
    for i, c in enumerate(conds):
        ax.text(i, 1.04, "n=%d" % pc[c]["support"], ha="center", fontsize=7)
    fig.tight_layout()
    out = os.path.join(out_dir, "per_condition_%s.png" % model_key)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)

    # 4) FDI (tooth-number) detection — does the model find the RIGHT teeth? Independent of disease.
    def fdi_num(key):
        m = re.match(r"\d+", str(key[1]))
        return int(m.group()) if m else None

    gold_fdi, tp_fdi, fp_fdi = Counter(), Counter(), Counter()
    for sid, g in gold.items():
        p = preds.get(sid, {})
        gk, pk = set(g), set(p)
        for key in gk:
            n = fdi_num(key)
            if n:
                gold_fdi[n] += 1
                if key in pk:
                    tp_fdi[n] += 1            # detected at the right (region, fdi)
        for key in pk - gk:
            n = fdi_num(key)
            if n:
                fp_fdi[n] += 1                # hallucinated tooth

    # Dental-chart layout (FDI), chart view: upper jaw on top, lower jaw on bottom.
    TOP = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    BOT = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
    recall = {n: (tp_fdi[n] / gold_fdi[n]) for n in (set(TOP) | set(BOT)) if gold_fdi[n]}
    grid = np.array([[recall.get(n, np.nan) for n in TOP],
                     [recall.get(n, np.nan) for n in BOT]])
    grid_m = np.ma.masked_invalid(grid)

    fig, ax = plt.subplots(figsize=(13, 3.4))
    cmap = plt.cm.RdYlGn.copy(); cmap.set_bad("lightgray")    # teeth absent in test = grey
    im = ax.imshow(grid_m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(16)); ax.set_xticklabels([str(n) for n in TOP])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["upper", "lower"])
    ax.set_title("%s - per-tooth (FDI) detection recall on test  (grey = not in test)" % model_key)
    for r, row in enumerate((TOP, BOT)):
        for c, n in enumerate(row):
            if gold_fdi[n]:
                ax.text(c, r - 0.16, str(n), ha="center", va="center", fontsize=7, fontweight="bold", color="black")
                ax.text(c, r + 0.18, "%.2f\n(n%d)" % (recall.get(n, 0), gold_fdi[n]),
                        ha="center", va="center", fontsize=6.5, color="black")
    ax.set_xticks([])   # each cell is labelled with its own FDI, so a shared x-axis would mislead
    fig.colorbar(im, fraction=0.025, pad=0.02, label="recall")
    fig.tight_layout()
    out = os.path.join(out_dir, "fdi_recall_chart_%s.png" % model_key)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("saved", out)

    # console table: worst-detected teeth first (most useful to eyeball)
    print("\nPer-FDI detection (worst recall first):")
    print("%-5s %8s %8s %8s %8s" % ("fdi", "gold", "found", "recall", "halluc"))
    for n in sorted(recall, key=lambda k: recall[k]):
        print("%-5d %8d %8d %8.3f %8d" % (n, gold_fdi[n], tp_fdi[n], recall[n], fp_fdi[n]))


if __name__ == "__main__":
    main()
