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
:root{--teal:#0e8a9c;--teal-d:#0b6c7b;--bg:#eef4f6;--card:#ffffff;--ink:#1b2b2f;--muted:#5b7178}
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink)}
header{background:linear-gradient(135deg,var(--teal),var(--teal-d));color:#fff;padding:1.1rem 1.4rem;
  display:flex;align-items:center;gap:.6rem;box-shadow:0 2px 8px rgba(0,0,0,.12)}
header .logo{font-size:1.6rem}
header h1{font-size:1.25rem;margin:0;font-weight:600}
header .sub{margin-left:auto;font-size:.8rem;opacity:.85}
.wrap{max-width:980px;margin:1.6rem auto;padding:0 1rem}
.card{background:var(--card);border-radius:14px;box-shadow:0 2px 12px rgba(11,108,123,.08);padding:1.4rem;margin-bottom:1.2rem}
.disclaimer{background:#fff7e6;border-left:4px solid #e0a423;color:#6b5108;padding:.7rem 1rem;border-radius:8px;
  font-size:.9rem;margin-bottom:1.2rem}
.uploadbox{border:2px dashed #b9d4da;border-radius:12px;padding:2.2rem;text-align:center;background:#f7fbfc}
.uploadbox p{color:var(--muted);margin:.4rem 0 1rem}
input[type=file]{margin:.6rem 0}
.btn{background:var(--teal);color:#fff;border:none;border-radius:9px;padding:.7rem 1.5rem;font-size:1rem;
  cursor:pointer;font-weight:600}
.btn:hover{background:var(--teal-d)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.xray{width:100%;border-radius:10px;border:1px solid #d7e3e6}
h2{font-size:1.05rem;color:var(--teal-d);margin:0 0 .6rem;border-bottom:2px solid #e3eef0;padding-bottom:.3rem}
.region{margin-bottom:1.1rem}
.region h3{font-size:.95rem;margin:.2rem 0 .4rem;color:var(--ink)}
.teeth{display:flex;flex-wrap:wrap;gap:.4rem;margin:.3rem 0}
.tooth{display:inline-flex;align-items:center;gap:.35rem;border-radius:20px;padding:.2rem .6rem;font-size:.82rem;
  border:1px solid #dde7ea;background:#f4f8f9}
.dot{width:.6rem;height:.6rem;border-radius:50%;display:inline-block}
.fdi{font-weight:700}
.note{font-size:.85rem;color:var(--muted);font-style:italic;margin:.2rem 0 0}
.summary{background:#eef6f7;border-left:4px solid var(--teal);padding:.8rem 1rem;border-radius:8px;margin-top:1rem}
.empty{color:#90a4aa;font-style:italic;font-size:.85rem}
details{margin-top:1rem}
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


def page(body_html, title="Dental Report Assistant"):
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><style>{STYLE}</style></head><body>"
        "<header><span class='logo'>&#129463;</span>"
        "<h1>Dental Report Assistant</h1>"
        "<span class='sub'>InternVL3-8B &middot; draft for dentist review</span></header>"
        f"<div class='wrap'>{body_html}</div></body></html>"
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
    # build the report column
    blocks = []
    regions = report.get("regions") if isinstance(report.get("regions"), dict) else None
    if regions:
        keys = [k for k in REGION_ORDER if k in regions] + [k for k in regions if k not in REGION_ORDER]
        items = [(REGION_NAMES.get(k, k), regions[k].get("teeth", []), regions[k].get("comment", ""))
                 for k in keys]
    else:
        items = [(REGION_NAMES.get(report.get("region", ""), "Region"),
                  report.get("teeth", []), report.get("comment", ""))]

    for label, teeth, comment in items:
        chips = []
        for t in teeth:
            code = t.get("condition", "")
            name = t.get("condition_name") or CONDITION_NAMES.get(code, code)
            color = _condition_color(code)
            chips.append(
                f"<span class='tooth'><span class='dot' style='background:{color}'></span>"
                f"<span class='fdi'>{esc(str(t.get('fdi','?')))}</span> {esc(name)}</span>"
            )
        teeth_html = ("<div class='teeth'>" + "".join(chips) + "</div>") if chips \
            else "<p class='empty'>No annotated teeth in this region.</p>"
        note = f"<p class='note'>{esc(comment)}</p>" if comment else ""
        blocks.append(f"<div class='region'><h3>{esc(label)}</h3>{teeth_html}{note}</div>")

    summary = report.get("summary")
    summary_html = f"<div class='summary'><strong>Summary.</strong> {esc(summary)}</div>" if summary else ""

    body = (
        "<div class='disclaimer'>Draft report &mdash; must be reviewed by a dentist.</div>"
        "<div class='grid'>"
        f"<div class='card'><h2>Panoramic X-ray</h2><img class='xray' src='{image_data_uri}'></div>"
        f"<div class='card'><h2>Report</h2>{''.join(blocks)}{summary_html}</div>"
        "</div>"
        "<div class='card'><details><summary>Show raw model output (JSON)</summary>"
        f"<pre>{esc(raw_text)}</pre></details>"
        "<a class='back' href='/'>&larr; Analyze another image</a></div>"
    )
    return page(body)


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
