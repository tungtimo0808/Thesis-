"""Final metrics on the TEST set for the selected checkpoint. Generates predictions (the only slow,"""
import json
import os
import sys

import config
from common import make_env, with_max_pixels, run_streaming, resolve_model_key


def chosen_checkpoint(model_key):
    if "--checkpoint" in sys.argv:
        return sys.argv[sys.argv.index("--checkpoint") + 1]
    marker = os.path.join(config.output_dir(model_key), "best_checkpoint.txt")
    if not os.path.exists(marker):
        sys.exit("No best_checkpoint.txt — run select_checkpoint.py %s first (or pass --checkpoint)." % model_key)
    return open(marker, encoding="utf-8").read().strip()


def main():
    model_key = resolve_model_key(sys.argv)
    m = config.MODELS[model_key]
    out_dir = config.output_dir(model_key)
    os.chdir(config.DATA_ROOT)

    ckpt = chosen_checkpoint(model_key)
    assert os.path.isdir(ckpt), "checkpoint not found: " + ckpt

    infer_out = os.path.join(ckpt, "infer_test.jsonl")
    pred_out = os.path.join(ckpt, "preds_test.jsonl")
    metrics_out = os.path.join(out_dir, "metrics_%s_test.json" % model_key)
    report_txt = os.path.join(out_dir, "report_%s_test.txt" % model_key)

    # Delete stale outputs — swift reuses an existing result file, so without this you would silently
    for stale in (infer_out, pred_out):
        if os.path.exists(stale):
            os.remove(stale)
            print("removed stale", stale)

    env = with_max_pixels(make_env(), m["uses_max_pixels"])

    print("\n=== EVAL %s | checkpoint: %s ===" % (model_key, ckpt))
    print(">>> Generating on the test set. tqdm / per-sample logs appear below; an initial silent")
    print(">>> minute is swift loading the model, not a hang.\n")
    run_streaming([
        "swift", "infer",
        "--model", m["model"],
        "--adapters", ckpt,
        "--val_dataset", config.TEST_JSONL,
        "--infer_backend", config.INFER_BACKEND,
        "--max_batch_size", str(config.INFER_MAX_BATCH_SIZE),
        "--max_new_tokens", str(config.INFER_MAX_NEW_TOKENS),
        "--temperature", "0",
        "--result_path", infer_out,
    ], env=env)

    run_streaming([sys.executable, config.ADAPTER_SCRIPT, "--val", config.TEST_JSONL,
                   "--swift-result", infer_out, "--out", pred_out])
    run_streaming([sys.executable, config.EVAL_SCRIPT, "--gold", config.TEST_JSONL, "--pred", pred_out,
                   "--out-json", metrics_out, "--tag", model_key + "/test"])

    # Human-readable report.
    metrics = json.load(open(metrics_out, encoding="utf-8"))
    lines = ["=" * 60, "PAN924 VLM test report (GCP L4, 4-bit QLoRA)",
             "model      : " + m["model"], "checkpoint : " + ckpt,
             "test set   : " + config.TEST_JSONL, "=" * 60, "", "-- Overall metrics --"]
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            lines.append("  %-24s = %.4f" % (name, value))
    lines.append("")
    lines.append("-- Per-condition (sorted by support) --")
    lines.append("%-6s %8s %8s %8s %8s" % ("cond", "support", "P", "R", "F1"))
    pc = metrics.get("per_condition", {})
    for c, d in sorted(pc.items(), key=lambda x: -x[1]["support"]):
        lines.append("%-6s %8d %8.3f %8.3f %8.3f" % (c, d["support"], d["P"], d["R"], d["F1"]))
    text = "\n".join(lines)
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("\n" + text)
    print("\nSaved:\n  JSON :", metrics_out, "\n  TXT  :", report_txt)
    print("Next: python visualize.py %s" % model_key)


if __name__ == "__main__":
    main()
