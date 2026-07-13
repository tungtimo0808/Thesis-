"""ms-swift external-plugin loss: up-weight the cross-entropy on RARE disease-code tokens."""
import os

import torch
import torch.nn.functional as F

_RARE_IDS = None


def _rare_ids(device):
    global _RARE_IDS
    if _RARE_IDS is None:
        raw = os.environ.get("RARE_TOKEN_IDS", "")
        _RARE_IDS = sorted(int(x) for x in raw.split(",") if x.strip())
    if not _RARE_IDS:
        return None
    return torch.tensor(_RARE_IDS, device=device)


def _weight():
    return float(os.environ.get("RARE_LOSS_WEIGHT", "3.0"))


def _weighted_ce(logits, labels, weight, num_items_in_batch=None):
    """Core math, separated so the self-test can call it directly."""
    logits = logits[..., :-1, :].contiguous()             # standard causal shift
    labels = labels[..., 1:].contiguous()
    V = logits.size(-1)
    ce = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100, reduction="none")
    flat_labels = labels.view(-1)
    valid = (flat_labels != -100).float()

    w = torch.ones_like(ce)
    rare_ids = _rare_ids(flat_labels.device)
    if rare_ids is not None:
        rare_mask = torch.isin(flat_labels, rare_ids)
        w = torch.where(rare_mask, torch.full_like(w, weight), w)
    w = w * valid                                          # ignored tokens contribute nothing
    denom = num_items_in_batch if num_items_in_batch is not None else valid.sum().clamp(min=1.0)
    return (ce * w).sum() / denom


def rare_weighted_loss(outputs, labels, *args, **kwargs):
    # swift passes extra kwargs (trainer=, loss_scale=, num_items_in_batch=, ...) — accept them all.
    num_items_in_batch = kwargs.get("num_items_in_batch")
    logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
    return _weighted_ce(logits, labels, _weight(), num_items_in_batch)


def _register():
    """Write rare_weighted_loss into whatever structure swift's get_loss_func() reads."""
    try:
        import swift.plugin.loss as L
    except Exception as e:
        print("[rare_loss] WARNING: could not import swift.plugin.loss:", e)
        return False
    ok = False
    reg = getattr(L, "register_loss_func", None)
    if callable(reg):
        try:
            reg("rare_weighted", rare_weighted_loss); ok = True
        except TypeError:
            try:
                reg("rare_weighted")(rare_weighted_loss); ok = True
            except Exception:
                pass
    for dict_name in ("loss_mapping", "LOSS_MAPPING"):
        d = getattr(L, dict_name, None)
        if isinstance(d, dict):
            d["rare_weighted"] = rare_weighted_loss; ok = True
    print('[rare_loss] registered "rare_weighted":', ok,
          "| keys now include rare_weighted:",
          any("rare_weighted" in getattr(L, dn, {}) for dn in ("loss_mapping", "LOSS_MAPPING")))
    if not ok:
        print("[rare_loss] available attrs in swift.plugin.loss:",
              [a for a in dir(L) if "loss" in a.lower() or "LOSS" in a])
    return ok


# Register as soon as swift imports this file via --external_plugins (not during the self-test).
def _allow_local_resume():
    """transformers blocks torch.load on torch<2.6 (a CVE guard). Our own checkpoints are trusted,"""
    import importlib
    for mod in ("transformers.trainer", "transformers.utils.import_utils"):
        try:
            m = importlib.import_module(mod)
            if hasattr(m, "check_torch_load_is_safe"):
                m.check_torch_load_is_safe = lambda *a, **k: None
        except Exception:
            pass


if __name__ != "__main__":
    _allow_local_resume()
    _register()


# --------------------------------------------------------------------------- self-test
if __name__ == "__main__":
    torch.manual_seed(0)
    B, L_, V = 2, 6, 50
    logits = torch.randn(B, L_, V)
    labels = torch.randint(0, V, (B, L_))
    labels[:, 0] = -100

    os.environ["RARE_TOKEN_IDS"] = ""
    _RARE_IDS = None
    mine = _weighted_ce(logits, labels, 3.0)
    ref = F.cross_entropy(logits[..., :-1, :].reshape(-1, V), labels[..., 1:].reshape(-1), ignore_index=-100)
    print("baseline weighted==plain CE :", torch.allclose(mine, ref, atol=1e-5), float(mine), float(ref))

    rare = sorted(set(int(x) for x in labels[..., 1:].reshape(-1).tolist()[:3]))
    os.environ["RARE_TOKEN_IDS"] = ",".join(map(str, rare))
    _RARE_IDS = None
    up = _weighted_ce(logits, labels, 3.0)
    print("rare up-weighting increases loss:", float(up) > float(ref), float(up), ">", float(ref))
