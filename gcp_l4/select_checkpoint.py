"""Pick the best checkpoint by LOWEST validation loss — free + instant (the loss at each step was"""
import json
import os
import sys

import config
from common import list_checkpoints, resolve_model_key


def main():
    model_key = resolve_model_key(sys.argv)
    out_dir = config.output_dir(model_key)
    ckpts = list_checkpoints(out_dir)
    if not ckpts:
        sys.exit("No checkpoints in %s — train first." % out_dir)

    # eval_steps == save_steps == 100, so each logged eval step lines up with a checkpoint-N folder.
    loss_by_step = {}
    state_path = os.path.join(ckpts[-1][1], "trainer_state.json")
    if os.path.exists(state_path):
        for e in json.load(open(state_path, encoding="utf-8")).get("log_history", []):
            if "eval_loss" in e and "step" in e:
                loss_by_step[int(e["step"])] = e["eval_loss"]

    path_by_step = {s: p for s, p in ckpts}
    if loss_by_step:
        ranked = sorted((loss, s) for s, loss in loss_by_step.items() if s in path_by_step)
        print("Checkpoints by val loss (lowest first):")
        for loss, s in ranked:
            print("   checkpoint-%-6d eval_loss = %.4f" % (s, loss))
        best = path_by_step[ranked[0][1]]
        print("\nBest (lowest val loss):", best, "-> eval_loss", round(ranked[0][0], 4))
    else:
        best = ckpts[-1][1]
        print("No eval_loss in trainer_state.json; falling back to the last checkpoint:", best)

    with open(os.path.join(out_dir, "best_checkpoint.txt"), "w", encoding="utf-8") as f:
        f.write(best + "\n")
    print("Wrote", os.path.join(out_dir, "best_checkpoint.txt"))
    print("Next: python evaluate.py %s" % model_key)


if __name__ == "__main__":
    main()
