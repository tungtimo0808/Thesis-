"""
Render a model's JSON dental report into a clean, human-readable report.

The fine-tuned VLMs output a JSON object (this is what the evaluation script scores).
For a user-facing tool, JSON is not nice to read, so this module turns that JSON into a
formatted report. Two output formats are provided:

  * "text" : plain text, good for a terminal or a .txt download.
  * "html" : a small self-contained HTML page, good for a web interface (e.g. FastAPI).

Both the full report (four regions + summary) and the single-region report are supported.

Typical use from a FastAPI endpoint:

    from render_report import render_report
    pretty = render_report(model_output_string, fmt="html")
    return HTMLResponse(pretty)

This file has no third-party dependencies; it only uses the Python standard library.
"""
import html
import json
import re


# --------------------------------------------------------------------------- label maps
# Condition code -> full name. The JSON usually carries "condition_name" already, but we
# keep this map so the report is still readable if that field is missing.
CONDITION_NAMES = {
    "H": "healthy",
    "C": "caries",
    "R": "restored",
    "Te": "endodontic treatment",
    "Im": "implant",
    "Rr": "residual root",
    "M3i": "impacted third molar",
    "M3f": "developing third molar",
    "CpuM": "prosthetic crown",
    "Dc": "crown destruction",
    "Di": "incisal or occlusal wear",
    "P": "pontic",
}

# Image-space region key -> readable name.
REGION_NAMES = {
    "image_upper_left": "Upper-left",
    "image_upper_right": "Upper-right",
    "image_lower_left": "Lower-left",
    "image_lower_right": "Lower-right",
}

# A fixed display order for the four regions in a full report.
REGION_ORDER = ["image_upper_right", "image_upper_left", "image_lower_left", "image_lower_right"]
RAW_REGION_ORDER = ["image_upper_left", "image_upper_right", "image_lower_left", "image_lower_right"]
REGION_KEYS = {"image_upper_left", "image_upper_right", "image_lower_left", "image_lower_right"}


# --------------------------------------------------------------------------- parsing
def _is_report_shape(obj):
    """True when a parsed object looks like one of the report schemas."""
    if not isinstance(obj, dict):
        return False
    if isinstance(obj.get("regions"), dict):
        return True
    if isinstance(obj.get("teeth"), list):
        return True
    return False


def _is_full_report_shape(obj):
    """True when a parsed object has the full four-region report schema."""
    if not isinstance(obj, dict):
        return False
    regions = obj.get("regions")
    if not isinstance(regions, dict):
        return False
    if not REGION_KEYS.issubset(regions):
        return False
    for key in REGION_KEYS:
        payload = regions.get(key)
        if not isinstance(payload, dict) or not isinstance(payload.get("teeth"), list):
            return False
    return isinstance(obj.get("summary"), str)


def _clean_prediction_text(prediction):
    prediction = prediction.strip()
    if prediction.startswith("```"):
        prediction = prediction.strip("`").strip()
    if prediction.startswith("json"):
        prediction = prediction[4:].strip()
    return prediction


def _repair_json_text(prediction):
    """Fix small JSON mistakes commonly made by the model."""
    # The model sometimes omits the comma between adjacent tooth objects:
    # [{"fdi": "35", ...} {"fdi": "36", ...}] or ...}{"fdi": "36"...}
    return re.sub(r"}\s*(?=\{\s*\")", "}, ", prediction)


def _insert_missing_comma(text, pos):
    """Insert one likely missing comma near a JSONDecodeError position."""
    window_start = max(0, pos - 20)
    window_end = min(len(text), pos + 80)
    for idx in range(window_start, window_end):
        if text[idx] not in "}]":
            continue
        next_idx = idx + 1
        while next_idx < len(text) and text[next_idx].isspace():
            next_idx += 1
        if next_idx < len(text) and text[next_idx] in '{"[':
            return text[:next_idx] + ", " + text[next_idx:]
    return text


def _loads_json_with_repairs(prediction):
    """Load JSON, repairing repeated missing-comma errors when possible."""
    text = _repair_json_text(prediction)
    for _ in range(20):
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            if "delimiter" not in exc.msg and "comma" not in exc.msg.lower():
                raise
            repaired = _insert_missing_comma(text, exc.pos)
            if repaired == text:
                raise
            text = repaired
    return json.loads(text)


def _parse_tooth_objects(text):
    teeth = []
    tooth_re = re.compile(
        r'\{\s*"fdi"\s*:\s*"([^"]+)"\s*,\s*"condition"\s*:\s*"([^"]+)"'
        r'\s*,\s*"condition_name"\s*:\s*"([^"]+)"\s*\}'
    )
    for match in tooth_re.finditer(text):
        teeth.append({
            "fdi": match.group(1),
            "condition": match.group(2),
            "condition_name": match.group(3),
        })
    return teeth


def _extract_string_after_key(text, key):
    match = re.search(r'"%s"\s*:\s*"((?:\\.|[^"\\])*)"' % re.escape(key), text)
    if not match:
        return ""
    try:
        return json.loads('"%s"' % match.group(1))
    except Exception:
        return match.group(1)


def _parse_full_prediction_loose(prediction):
    """Best-effort parser for malformed but schema-like model output."""
    regions = {}
    for index, key in enumerate(RAW_REGION_ORDER):
        start_marker = '"%s"' % key
        start = prediction.find(start_marker)
        if start < 0:
            return None
        next_positions = [
            prediction.find('"%s"' % next_key, start + len(start_marker))
            for next_key in RAW_REGION_ORDER[index + 1:]
        ]
        next_positions.append(prediction.find('"summary"', start + len(start_marker)))
        next_positions = [pos for pos in next_positions if pos >= 0]
        end = min(next_positions) if next_positions else len(prediction)
        chunk = prediction[start:end]
        teeth_match = re.search(r'"teeth"\s*:\s*\[(.*?)\]\s*,\s*"comment"', chunk, re.S)
        teeth_text = teeth_match.group(1) if teeth_match else chunk
        regions[key] = {
            "teeth": _parse_tooth_objects(teeth_text),
            "comment": _extract_string_after_key(chunk, "comment"),
        }
    summary = _extract_string_after_key(prediction, "summary")
    report = {"regions": regions, "summary": summary}
    return report if _is_full_report_shape(report) else None


def parse_full_prediction(prediction):
    """Parse only a full four-region report; reject nested/single-region objects."""
    if isinstance(prediction, dict):
        return prediction if _is_full_report_shape(prediction) else None
    if not isinstance(prediction, str):
        return None
    prediction = _clean_prediction_text(prediction)
    candidates = [prediction, _repair_json_text(prediction)]
    for candidate in candidates:
        try:
            obj = _loads_json_with_repairs(candidate)
            return obj if _is_full_report_shape(obj) else None
        except Exception:
            pass
    loose = _parse_full_prediction_loose(prediction)
    if loose is not None:
        return loose
    try:
        start = prediction.find("{")
        if start >= 0:
            obj, _ = json.JSONDecoder().raw_decode(_repair_json_text(prediction[start:]))
            return obj if _is_full_report_shape(obj) else None
    except Exception:
        try:
            obj, _ = json.JSONDecoder().raw_decode(prediction[start:])
            return obj if _is_full_report_shape(obj) else None
        except Exception:
            return None


def parse_prediction(prediction):
    """Accept a dict or a JSON string and return a dict, or None if it cannot be parsed.

    The model sometimes wraps the JSON in extra text, markdown fences, or explanatory
    text after the JSON. As a fallback, read the first complete JSON object we can find.
    """
    if isinstance(prediction, dict):
        return prediction if _is_report_shape(prediction) else None
    if not isinstance(prediction, str):
        return None
    prediction = _clean_prediction_text(prediction)
    candidates = [prediction, _repair_json_text(prediction)]
    for candidate in candidates:
        try:
            obj = _loads_json_with_repairs(candidate)
            return obj if _is_report_shape(obj) else None
        except Exception:
            pass
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for start, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(candidate[start:])
                if _is_report_shape(obj):
                    return obj
            except Exception:
                pass
    start, end = prediction.find("{"), prediction.rfind("}")
    if 0 <= start < end:
        try:
            obj = _loads_json_with_repairs(prediction[start:end + 1])
            return obj if _is_report_shape(obj) else None
        except Exception:
            pass
    return None


def parse_error(prediction):
    """Return a short JSON parse error for debugging display."""
    if not isinstance(prediction, str):
        return "Model output is not text."
    try:
        obj = _loads_json_with_repairs(_clean_prediction_text(prediction))
        if isinstance(obj, dict) and isinstance(obj.get("regions"), dict):
            missing = sorted(REGION_KEYS.difference(obj["regions"]))
            if missing:
                return "JSON is valid but the full report is missing region keys: %s" % ", ".join(missing)
            if not isinstance(obj.get("summary"), str):
                return "JSON is valid but the full report is missing a string summary."
            if _is_full_report_shape(obj):
                return ""
        return "JSON is valid but it does not match the full four-region report schema."
    except json.JSONDecodeError as exc:
        return "JSON error at line %s column %s: %s" % (exc.lineno, exc.colno, exc.msg)
    except Exception as exc:
        return str(exc)


def _condition_name(tooth):
    """Best-effort condition name for one tooth entry."""
    name = tooth.get("condition_name")
    if name:
        return str(name)
    code = tooth.get("condition", "")
    return CONDITION_NAMES.get(code, code or "unknown")


def _iter_regions(report):
    """Yield (region_label, teeth_list, comment) for either report schema.

    Full report  -> {"regions": {<key>: {"teeth": [...], "comment": ...}}, "summary": ...}
    Regional one -> {"region": <key>, "teeth": [...], "comment": ...}
    """
    if isinstance(report.get("regions"), dict):
        regions = report["regions"]
        keys = [k for k in REGION_ORDER if k in regions] + \
               [k for k in regions if k not in REGION_ORDER]
        for key in keys:
            payload = regions.get(key) or {}
            label = REGION_NAMES.get(key, key)
            yield label, payload.get("teeth", []), payload.get("comment", "")
    elif "teeth" in report:
        key = report.get("region", "")
        label = REGION_NAMES.get(key, key or "Region")
        yield label, report.get("teeth", []), report.get("comment", "")


# --------------------------------------------------------------------------- text output
def render_text(report):
    """Render the report as plain text."""
    lines = []
    lines.append("=" * 48)
    lines.append("DENTAL RADIOGRAPH REPORT".center(48))
    lines.append("=" * 48)
    lines.append("")

    for label, teeth, comment in _iter_regions(report):
        lines.append(label)
        lines.append("-" * len(label))
        if teeth:
            for t in teeth:
                fdi = t.get("fdi", "?")
                lines.append("  Tooth %-4s %s" % (fdi, _condition_name(t)))
        else:
            lines.append("  No annotated teeth in this region.")
        if comment:
            lines.append("  Note: %s" % comment)
        lines.append("")

    summary = report.get("summary")
    if summary:
        lines.append("Summary")
        lines.append("-------")
        lines.append(summary)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- html output
def render_html(report, title="Dental Radiograph Report"):
    """Render the report as a small self-contained HTML page."""
    esc = html.escape
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append("<title>%s</title>" % esc(title))
    parts.append(
        "<style>"
        "body{font-family:system-ui,Arial,sans-serif;max-width:760px;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a;line-height:1.5}"
        "h1{font-size:1.5rem;border-bottom:2px solid #333;padding-bottom:.3rem}"
        "h2{font-size:1.1rem;margin-top:1.6rem;color:#0b4f6c}"
        "table{border-collapse:collapse;width:100%;margin:.4rem 0}"
        "th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;font-size:.95rem}"
        "th{background:#f2f6f8}"
        ".note{font-style:italic;color:#444;margin:.3rem 0 .2rem}"
        ".summary{background:#f7f7f4;border-left:4px solid #0b4f6c;padding:.6rem .8rem;"
        "margin-top:1.4rem}"
        ".empty{color:#777;font-style:italic}"
        "</style></head><body>"
    )
    parts.append("<h1>%s</h1>" % esc(title))

    for label, teeth, comment in _iter_regions(report):
        parts.append("<h2>%s</h2>" % esc(label))
        if teeth:
            parts.append("<table><tr><th>Tooth (FDI)</th><th>Condition</th></tr>")
            for t in teeth:
                parts.append("<tr><td>%s</td><td>%s</td></tr>" %
                             (esc(str(t.get("fdi", "?"))), esc(_condition_name(t))))
            parts.append("</table>")
        else:
            parts.append("<p class='empty'>No annotated teeth in this region.</p>")
        if comment:
            parts.append("<p class='note'>%s</p>" % esc(str(comment)))

    summary = report.get("summary")
    if summary:
        parts.append("<div class='summary'><strong>Summary.</strong> %s</div>" % esc(str(summary)))

    parts.append("</body></html>")
    return "".join(parts)


# --------------------------------------------------------------------------- top-level
def render_report(prediction, fmt="text", title="Dental Radiograph Report"):
    """Turn a model prediction (dict or JSON string) into a pretty report.

    fmt = "text" or "html". Raises ValueError if the prediction is not valid JSON.
    """
    report = parse_prediction(prediction)
    if report is None:
        raise ValueError("The model output is not valid JSON and could not be rendered.")
    if fmt == "html":
        return render_html(report, title=title)
    if fmt == "text":
        return render_text(report)
    raise ValueError("Unknown format %r (use 'text' or 'html')." % fmt)


if __name__ == "__main__":
    # Small demo with a sample report.
    sample = {
        "regions": {
            "image_lower_left": {
                "teeth": [
                    {"fdi": "33", "condition": "R", "condition_name": "restored"},
                    {"fdi": "43", "condition": "R", "condition_name": "restored"},
                ],
                "comment": "This region shows restorations on teeth 33 and 43.",
            }
        },
        "summary": "The lower-left region has two restored teeth; the rest are healthy.",
    }
    print(render_text(sample))
