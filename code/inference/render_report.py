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


# --------------------------------------------------------------------------- parsing
def parse_prediction(prediction):
    """Accept a dict or a JSON string and return a dict, or None if it cannot be parsed.

    The model sometimes wraps the JSON in extra text, so as a fallback we try to read the
    object between the first '{' and the last '}'.
    """
    if isinstance(prediction, dict):
        return prediction
    if not isinstance(prediction, str):
        return None
    try:
        return json.loads(prediction)
    except Exception:
        start, end = prediction.find("{"), prediction.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(prediction[start:end + 1])
            except Exception:
                return None
        return None


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
