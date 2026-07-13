"""Shared condition labels for the PAN924 dental report pipeline.

This file is the small, auditable place to point to when asked how raw tooth
condition labels were merged into the 12 final classes used in the slides.
"""

from __future__ import annotations

CONDITION_NAMES: dict[str, str] = {
    "H": "healthy",
    "R": "restored",
    "Te": "endodontic treatment",
    "CpuM": "prosthetic crown",
    "M3i": "impacted third molar",
    "M3f": "developing third molar",
    "Di": "incisal or occlusal wear",
    "C": "caries",
    "Rr": "residual root",
    "P": "pontic",
    "Im": "implant",
    "Dc": "crown destruction",
}

FINAL_CONDITION_CLASSES: list[str] = list(CONDITION_NAMES)

# Raw label -> final label. These are the merge rules shown in the slides.
CONDITION_REMAP: dict[str, str] = {
    "Ri": "Te",
    "RiM": "Te",
    "TeM": "Te",
    "I": "M3i",
}

RARE_CONDITION_CLASSES: set[str] = {"Dc", "Im", "P", "Rr", "M3f"}


def normalize_condition(raw_label: object) -> str | None:
    """Return the final 12-class condition code, or None for unknown labels."""
    if raw_label is None:
        return None
    label = str(raw_label).strip()
    if not label:
        return None
    label = CONDITION_REMAP.get(label, label)
    if label not in CONDITION_NAMES:
        return None
    return label


def condition_name(code: str) -> str:
    """Human-readable condition name for report targets."""
    return CONDITION_NAMES.get(code, code)
