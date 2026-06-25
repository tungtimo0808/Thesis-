"""Rebalance rare classes with IMAGE AUGMENTATION (smarter than plain duplication).

oversample.py duplicates a rare report's row N times — identical pixels, so the model can memorise
them (overfit). This script instead keeps 1 original + creates (N-1) photometrically AUGMENTED copies
of the image, each with a fresh file but the SAME report. The rare disease is now seen under varied
exposure/rotation, which generalises instead of memorising.

SAFE augmentations for dental panoramic crops (preserve FDI laterality + the condition):
  brightness, contrast, gamma, mild gaussian noise, small rotation (<=4 deg).
NEVER horizontal flip — it swaps left/right quadrants, so tooth 16 would look like 26 (wrong FDI).

Replication factor per report = same rule as oversample.py (driven by the rarest condition it holds,
capped at REPLICATION_CAP). Writes train_balanced.jsonl, which train.py uses automatically.

Usage:
  python augment_rare.py            # needs Pillow; falls back is just: use oversample.py instead
"""
import io
import json
import math
import os
import random
from collections import Counter

from PIL import Image, ImageEnhance

import config
from oversample import flatten_conditions, gold_text

AUG_DIR_REL = "vlm_report_dataset/common/images_aug"     # relative to DATA_ROOT
RNG = random.Random(config.SEED)


def augment_image(img):
    """Apply a random SAFE photometric/geometric jitter. Returns a new PIL image."""
    img = img.convert("RGB")
    # brightness / contrast / gamma-ish (all keep anatomy + laterality intact)
    img = ImageEnhance.Brightness(img).enhance(RNG.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(RNG.uniform(0.85, 1.15))
    img = ImageEnhance.Sharpness(img).enhance(RNG.uniform(0.8, 1.3))
    # small rotation only (positioning varies in real panoramics); black fill = X-ray background
    angle = RNG.uniform(-4, 4)
    if abs(angle) > 0.3:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    return img


def main():
    src = os.path.join(config.DATA_ROOT, config.TRAIN_JSONL)
    dst = os.path.join(config.DATA_ROOT, config.TRAIN_BALANCED_JSONL)
    aug_root = os.path.join(config.DATA_ROOT, AUG_DIR_REL)
    os.makedirs(aug_root, exist_ok=True)
    assert os.path.exists(src), "train.jsonl not found at " + src

    rows = [json.loads(ln) for ln in open(src, encoding="utf-8") if ln.strip()]

    # 1) per-condition counts -> per-condition factor (same logic as oversample.py)
    conds_per_row, total = [], Counter()
    for o in rows:
        try:
            conds = flatten_conditions(json.loads(gold_text(o)))
        except Exception:
            conds = []
        conds_per_row.append(conds)
        total.update(conds)
    factor = {c: max(1, math.ceil(config.OVERSAMPLE_TARGET / n)) for c, n in total.items()}

    out_rows = []
    made = 0
    for ri, (o, conds) in enumerate(zip(rows, conds_per_row)):
        out_rows.append(o)                                   # always keep the original
        rep = 1 if not conds else min(config.REPLICATION_CAP, max(factor[c] for c in set(conds)))
        for k in range(rep - 1):                             # rep-1 AUGMENTED variants
            new = json.loads(json.dumps(o))                  # deep copy
            new["id"] = "%s_aug%d" % (o.get("id", "r%d" % ri), k)
            new_imgs = []
            for img_rel in o.get("images", []):
                src_img = os.path.join(config.DATA_ROOT, img_rel)
                if not os.path.exists(src_img):
                    new_imgs.append(img_rel)                  # missing source: keep original ref
                    continue
                stem = os.path.splitext(os.path.basename(img_rel))[0]
                out_name = "%s__%s_aug%d.jpg" % (os.path.basename(os.path.dirname(img_rel)), stem, k)
                out_rel = AUG_DIR_REL + "/" + out_name
                out_abs = os.path.join(config.DATA_ROOT, out_rel)
                if not os.path.exists(out_abs):
                    try:
                        with Image.open(src_img) as im:
                            augment_image(im).save(out_abs, quality=92)
                        made += 1
                    except Exception as e:
                        print("  [warn] could not augment", src_img, "-", e)
                        out_rel = img_rel
                new_imgs.append(out_rel)
            new["images"] = new_imgs
            out_rows.append(new)

    with open(dst, "w", encoding="utf-8") as f:
        for o in out_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("Reports: %d -> %d  (augmented images created: %d, in %s)" %
          (len(rows), len(out_rows), made, AUG_DIR_REL))
    print("Wrote", dst, "\ntrain.py uses it automatically (config.USE_BALANCED_TRAIN = True).")


if __name__ == "__main__":
    main()
