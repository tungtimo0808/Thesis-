"""
Dental Report Assistant — a FastAPI web interface.

Upload a panoramic dental X-ray and get a clean, readable report. Behind the scenes the
best model from the thesis (InternVL3-8B fine-tuned with QLoRA) reads the image, returns a
structured JSON report, and this app renders it as a dentist-friendly page.

The model output stays JSON internally (so it is still measurable against the gold
reports); the JSON is only turned into a nice report for display.

Run (on the GPU machine, inside the project venv):
    pip install fastapi "uvicorn[standard]" python-multipart
    cd ~/pan924              # the folder that holds vlm_report_dataset/ (for relative paths)
    uvicorn app:app --host 0.0.0.0 --port 8000
    # then open http://<server>:8000

The model is loaded lazily on the first request (about a minute), then reused.
"""
import base64
import datetime
import hashlib
import os
import tempfile

os.environ.setdefault("MAX_PIXELS", "401408")   # InternVL3 reads this; matches training
os.environ.setdefault("USE_HF", "1")            # download from Hugging Face, not ModelScope

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

from render_report import parse_prediction, CONDITION_NAMES, REGION_NAMES, REGION_ORDER


# --------------------------------------------------------------------------- model config
MODEL = "OpenGVLab/InternVL3-8B"
ADAPTER = os.path.expanduser("~/pan924_runs/internvl/checkpoint-500")

# The exact full-report prompt used in training (see the thesis appendix).
FULL_REPORT_PROMPT = (
    "<image>\n"
    "This is a full panoramic dental radiograph.\n"
    "Divide the image into four image-space regions: image_upper_left, image_upper_right, "
    "image_lower_left, and image_lower_right.\n"
    "For each region, list the visible annotated teeth using FDI notation and report each "
    "tooth condition.\n"
    "Return only valid JSON with this schema:\n"
    '{"regions":{"image_upper_left":{"teeth":[{"fdi":"string","condition":"string",'
    '"condition_name":"string"}],"comment":"string"},"image_upper_right":{"teeth":[{"fdi":'
    '"string","condition":"string","condition_name":"string"}],"comment":"string"},'
    '"image_lower_left":{"teeth":[{"fdi":"string","condition":"string","condition_name":'
    '"string"}],"comment":"string"},"image_lower_right":{"teeth":[{"fdi":"string",'
    '"condition":"string","condition_name":"string"}],"comment":"string"}},"summary":"string"}'
)

# Conditions that are abnormal (shown in a warm colour so they stand out).
ABNORMAL = {"C", "Te", "Im", "Rr", "M3i", "M3f", "CpuM", "Dc", "Di", "P"}

# A panoramic X-ray is a mirrored view, so the image-space region key maps to the opposite
# anatomical side. This fallback is only used when a region has no teeth to read the FDI from.
QUADRANT_OF_KEY = {
    "image_upper_left": 1,    # patient's upper right (FDI 11-18)
    "image_upper_right": 2,   # patient's upper left  (FDI 21-28)
    "image_lower_left": 4,    # patient's lower right (FDI 41-48)
    "image_lower_right": 3,   # patient's lower left  (FDI 31-38)
}
QUADRANT_NAME = {1: "Upper right", 2: "Upper left", 3: "Lower left", 4: "Lower right"}


def _quadrant_of(region_key, teeth):
    """Anatomical quadrant (1-4) of a region, read from the FDI numbers when possible."""
    digits = [str(t.get("fdi", "")).strip()[:1] for t in teeth]
    digits = [d for d in digits if d in "1234"]
    if digits:
        return int(max(set(digits), key=digits.count))   # the most common quadrant present
    return QUADRANT_OF_KEY.get(region_key, 9)

app = FastAPI(title="Dental Report Assistant")
_engine = None   # loaded on first use


def get_engine():
    """Build the inference engine once and reuse it."""
    global _engine
    if _engine is None:
        import torch
        from swift.llm import PtEngine
        _engine = PtEngine(MODEL, torch_dtype=torch.bfloat16, adapters=[ADAPTER], max_batch_size=1)
    return _engine


def run_model(image_path):
    """Run InternVL3 on one image and return the raw JSON string."""
    from swift.llm import InferRequest, RequestConfig
    engine = get_engine()
    request = InferRequest(
        messages=[{"role": "user", "content": FULL_REPORT_PROMPT}],
        images=[image_path],
    )
    config = RequestConfig(max_tokens=1280, temperature=0.0)
    responses = engine.infer([request], config)
    return responses[0].choices[0].message.content


# --------------------------------------------------------------------------- styling
STYLE = """
:root{--teal:#0e8a9c;--teal-d:#0b6c7b;--bg:#eef4f6;--card:#ffffff;--ink:#1b2b2f;--muted:#5b7178;--rule:#d7e3e6}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(135deg,var(--teal),var(--teal-d));color:#fff;padding:1.1rem 1.4rem;
  display:flex;align-items:center;gap:.6rem;box-shadow:0 2px 8px rgba(0,0,0,.12)}
header .logo{font-size:1.6rem}
header h1{font-size:1.25rem;margin:0;font-weight:600}
header .sub{margin-left:auto;font-size:.8rem;opacity:.85}
.wrap{max-width:880px;margin:1.6rem auto;padding:0 1rem}
.card{background:var(--card);border-radius:14px;box-shadow:0 2px 12px rgba(11,108,123,.08);padding:1.4rem;margin-bottom:1.2rem}
.disclaimer{background:#fff7e6;border-left:4px solid #e0a423;color:#6b5108;padding:.7rem 1rem;border-radius:8px;
  font-size:.9rem;margin-bottom:1.2rem}
.uploadbox{border:2px dashed #b9d4da;border-radius:12px;padding:2.2rem;text-align:center;background:#f7fbfc}
.uploadbox p{color:var(--muted);margin:.4rem 0 1rem}
input[type=file]{margin:.6rem 0}
.btn{background:var(--teal);color:#fff;border:none;border-radius:9px;padding:.7rem 1.5rem;font-size:1rem;
  cursor:pointer;font-weight:600}
.btn:hover{background:var(--teal-d)}

/* ---- results: X-ray on the left, a compact report on the right ---- */
.wrap.wide{max-width:1040px}
.grid{display:grid;grid-template-columns:0.95fr 1.05fr;gap:1.2rem;align-items:start}
@media(max-width:780px){.grid{grid-template-columns:1fr}}
.xray{width:100%;border-radius:10px;border:1px solid var(--rule)}
.rmeta{margin-top:.75rem;display:flex;flex-wrap:wrap;gap:.2rem .9rem;font-size:.8rem;color:var(--muted)}
.rmeta b{color:var(--teal-d);font-weight:700;font-variant:small-caps;letter-spacing:.3px;margin-right:.25rem}
h2{font-size:1.05rem;color:var(--teal-d);margin:0 0 .7rem;border-bottom:2px solid #e3eef0;padding-bottom:.3rem}
.region{margin-bottom:.85rem;padding-bottom:.65rem;border-bottom:1px solid #eef3f4}
.region:last-of-type{border-bottom:none;margin-bottom:.4rem}
.region h3{font-size:.92rem;margin:.1rem 0 .3rem;color:var(--ink)}
.qsub{font-weight:400;font-size:.72rem;color:var(--muted);font-variant:small-caps;letter-spacing:.3px;margin-left:.35rem}
.note{font-size:.88rem;color:#33474c;line-height:1.5;margin:.15rem 0 .4rem}
.teethlist{list-style:none;display:flex;flex-wrap:wrap;gap:.3rem .9rem;margin:.2rem 0 0;padding:0;font-size:.82rem}
.teethlist li{display:inline-flex;align-items:center;gap:.35rem;color:#2a3b40}
.teethlist li.abn{color:#b23b37;font-weight:600}
.teethlist .fdi{font-weight:700}
.dot{width:.58rem;height:.58rem;border-radius:50%;display:inline-block;flex:none}
.impression{background:#eef6f7;border-left:4px solid var(--teal);padding:.7rem .9rem;border-radius:8px;
  margin-top:1rem;font-size:.9rem;line-height:1.55}
.empty{color:#90a4aa;font-style:italic;font-size:.85rem}
details{margin-top:.4rem}
details summary{cursor:pointer;color:var(--muted);font-size:.85rem}
pre{background:#0f1f23;color:#cfe8ec;padding:.8rem;border-radius:8px;overflow:auto;font-size:.78rem}
a.back{display:inline-block;margin-top:1rem;color:var(--teal-d);text-decoration:none;font-weight:600}
"""


def _condition_color(code):
    if code == "H":
        return "#2e9e5b"        # healthy = green
    if code == "R":
        return "#2f7dd1"        # restored = blue
    if code in ABNORMAL:
        return "#d9534f"        # abnormal = red
    return "#90a4aa"            # unknown = grey


def page(body_html, title="Dental Report Assistant", wrap_class="wrap"):
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><style>{STYLE}</style></head><body>"
        "<header><span class='logo'>&#129463;</span>"
        "<h1>Dental Report Assistant</h1>"
        "<span class='sub'>InternVL3-8B &middot; draft for dentist review</span></header>"
        f"<div class='{wrap_class}'>{body_html}</div></body></html>"
    )


def upload_page():
    body = (
        "<div class='disclaimer'>This tool produces a <strong>draft</strong> report from a "
        "panoramic X-ray. A qualified dentist must review and confirm every finding before "
        "any clinical use.</div>"
        "<div class='card'>"
        "<form action='/analyze' method='post' enctype='multipart/form-data'>"
        "<div class='uploadbox'>"
        "<h2 style='border:none'>Upload a panoramic X-ray</h2>"
        "<p>JPG or PNG. The model lists the teeth (FDI) and their condition per region.</p>"
        "<input type='file' name='file' accept='image/*' required><br>"
        "<button class='btn' type='submit'>Generate report</button>"
        "</div></form></div>"
    )
    return page(body)


def results_page(report, image_data_uri, raw_text):
    esc = lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # ---- collect the regions as (quadrant_number, teeth, comment) and order them 1->4
    sections = []
    regions = report.get("regions") if isinstance(report.get("regions"), dict) else None
    if regions:
        for key, payload in regions.items():
            payload = payload or {}
            teeth = payload.get("teeth", []) or []
            sections.append((_quadrant_of(key, teeth), teeth, payload.get("comment", "")))
    else:
        teeth = report.get("teeth", []) or []
        sections.append((_quadrant_of(report.get("region", ""), teeth),
                         teeth, report.get("comment", "")))
    sections.sort(key=lambda s: s[0])

    # ---- one compact findings block per quadrant: a narrative line + a tooth list
    blocks = []
    for quadrant, teeth, comment in sections:
        qname = QUADRANT_NAME.get(quadrant, "Other region")
        note = f"<p class='note'>{esc(comment)}</p>" if comment else ""
        if teeth:
            items = []
            for t in teeth:
                code = t.get("condition", "")
                name = t.get("condition_name") or CONDITION_NAMES.get(code, code)
                cls = " class='abn'" if code in ABNORMAL else ""
                items.append(
                    f"<li{cls}><span class='dot' style='background:{_condition_color(code)}'></span>"
                    f"<span class='fdi'>{esc(str(t.get('fdi', '?')))}</span> {esc(name)}</li>"
                )
            teeth_html = "<ul class='teethlist'>" + "".join(items) + "</ul>"
        else:
            teeth_html = "<p class='empty'>No annotated teeth in this region.</p>"
        blocks.append(
            f"<div class='region'><h3>{esc(qname)}<span class='qsub'>quadrant {quadrant}</span></h3>"
            f"{note}{teeth_html}</div>"
        )

    summary = report.get("summary")
    impression = (f"<div class='impression'><strong>Impression.</strong> {esc(summary)}</div>"
                  if summary else "")

    date_str = datetime.date.today().strftime("%d %B %Y")
    ref = "OPG-" + hashlib.sha1(image_data_uri.encode()).hexdigest()[:6].upper()

    body = (
        "<div class='disclaimer'>This is an <strong>AI-generated draft report</strong>. "
        "A qualified dentist must review and confirm every finding before any clinical use.</div>"
        "<div class='grid'>"
            "<div class='card'>"
                "<h2>Panoramic X-ray</h2>"
                f"<img class='xray' src='{image_data_uri}'>"
                "<div class='rmeta'>"
                    f"<span><b>Report ID</b>{ref}</span>"
                    f"<span><b>Date</b>{date_str}</span>"
                    "<span><b>Modality</b>OPG</span>"
                "</div>"
            "</div>"
            "<div class='card'>"
                "<h2>Report</h2>"
                + "".join(blocks)
                + impression
            + "</div>"
        "</div>"
        "<div class='card'><details><summary>Show raw model output (JSON)</summary>"
        f"<pre>{esc(raw_text)}</pre></details>"
        "<a class='back' href='/'>&larr; Analyze another image</a></div>"
    )
    return page(body, wrap_class="wrap wide")


# --------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def home():
    return upload_page()


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".png"
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        raw_text = run_model(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    report = parse_prediction(raw_text)
    if report is None:
        body = ("<div class='card'><h2>Could not read the report</h2>"
                "<p>The model did not return valid JSON.</p>"
                f"<pre>{raw_text}</pre><a class='back' href='/'>&larr; Try again</a></div>")
        return page(body)

    mime = "image/png" if suffix.lower().endswith("png") else "image/jpeg"
    image_data_uri = "data:%s;base64,%s" % (mime, base64.b64encode(data).decode())
    return results_page(report, image_data_uri, raw_text)
