"""Zero-shot baseline: evaluate the BASE model (no LoRA adapter) on the TEST set."""
import json
import os
import sys

import config
from common import make_env, with_max_pixels, run_streaming, resolve_model_key


def main():
    model_key = resolve_model_key(sys.argv)
    m = config.MODELS[model_key]
    out_dir = config.output_dir(model_key)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(config.DATA_ROOT)   # image paths in the jsonl are relative to DATA_ROOT

    infer_out = os.path.join(out_dir, "infer_test_base.jsonl")
    pred_out = os.path.join(out_dir, "preds_test_base.jsonl")
    metrics_out = os.path.join(out_dir, "metrics_%s_base.json" % model_key)
    report_txt = os.path.join(out_dir, "report_%s_base.txt" % model_key)

    # swift reuses an existing result file, so delete stale ones to avoid re-scoring old output.
    for stale in (infer_out, pred_out):
        if os.path.exists(stale):
            os.remove(stale)
            print("removed stale", stale)

    env = with_max_pixels(make_env(), m["uses_max_pixels"])
    # The base model has not learned to stop early, so it tends to generate up to the full
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    base_max_batch = 2

    # The only difference from evaluate.py: no "--adapters", so the base model runs as-is.
    cmd = [
        "swift", "infer",
        "--model", m["model"],
        "--val_dataset", config.TEST_JSONL,
        "--torch_dtype", config.TORCH_DTYPE,
        "--quant_method", config.QUANT_METHOD,
        "--quant_bits", str(config.QUANT_BITS),
        "--bnb_4bit_quant_type", config.BNB_4BIT_QUANT_TYPE,
        "--bnb_4bit_use_double_quant", config.BNB_4BIT_USE_DOUBLE_QUANT,
        "--bnb_4bit_compute_dtype", config.BNB_4BIT_COMPUTE_DTYPE,
        "--infer_backend", config.INFER_BACKEND,
        "--max_batch_size", str(base_max_batch),
        "--max_new_tokens", str(config.INFER_MAX_NEW_TOKENS),
        "--temperature", "0",
        "--result_path", infer_out,
    ]
    if m["attn_impl"] is not None:
        cmd += ["--attn_impl", m["attn_impl"]]   # Phi-3.5-vision needs eager attention

    print("\n=== BASE EVAL %s | model: %s (no adapter) ===" % (model_key, m["model"]))
    print(">>> Generating on the test set. The first silent minute is swift loading the model.\n")
    run_streaming(cmd, env=env)

    run_streaming([sys.executable, config.ADAPTER_SCRIPT, "--val", config.TEST_JSONL,
                   "--swift-result", infer_out, "--out", pred_out])
    run_streaming([sys.executable, config.EVAL_SCRIPT, "--gold", config.TEST_JSONL, "--pred", pred_out,
                   "--out-json", metrics_out, "--tag", model_key + "/base"])

    # Human-readable report (same layout as evaluate.py).
    metrics = json.load(open(metrics_out, encoding="utf-8"))
    lines = ["=" * 60, "PAN924 VLM test report - BASE MODEL (zero-shot, no fine-tuning)",
             "model      : " + m["model"], "adapter    : none (base model)",
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
    print("Compare against the fine-tuned report: report_%s_test.txt" % model_key)


if __name__ == "__main__":
    main()
