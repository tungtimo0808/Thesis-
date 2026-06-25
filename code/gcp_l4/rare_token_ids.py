"""Compute the tokenizer ids that are DISTINCTIVE to rare disease codes, for the loss-up-weighting
plugin (rare_loss.py).

"Distinctive" = an id that shows up when encoding a rare code (in JSON-ish contexts) but NOT when
encoding the common codes (H, R). Subtracting the common-code ids drops the shared quote/colon/space
tokens, so we only up-weight the subword(s) that actually carry the rare code — not punctuation.

This is a heuristic (BPE can merge a code with neighbouring quotes); it is good enough to bias the
model toward emitting rare codes, and it is opt-in. train.py calls compute_rare_token_ids() and
passes the result to swift via the RARE_TOKEN_IDS env var.
"""
CONTEXTS = ['"%s"', '"condition": "%s"', '"condition":"%s"', ' %s', '%s']


def _ids_for(tok, codes):
    ids = set()
    for c in codes:
        for ctx in CONTEXTS:
            ids.update(tok.encode(ctx % c, add_special_tokens=False))
    return ids


def compute_rare_token_ids(model_name, rare_codes, common_codes):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    rare = _ids_for(tok, rare_codes)
    common = _ids_for(tok, common_codes)
    distinctive = sorted(rare - common)
    return distinctive, tok


if __name__ == "__main__":
    import sys
    import config
    model = sys.argv[1] if len(sys.argv) > 1 else config.MODELS["qwen"]["model"]
    ids, tok = compute_rare_token_ids(model, config.RARE_CODES, config.COMMON_CODES)
    print("model:", model)
    print("rare codes:", config.RARE_CODES, "| common:", config.COMMON_CODES)
    print("distinctive rare token ids (%d):" % len(ids), ids)
    print("decoded:", [tok.decode([i]) for i in ids])
