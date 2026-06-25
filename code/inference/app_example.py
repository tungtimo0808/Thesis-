"""
Minimal FastAPI example that serves a pretty dental report.

This is a starting point for the user-facing interface. It does NOT run a model; it takes
a model's JSON output and returns a formatted report. Plug your own model call into the
`/report` endpoint where noted.

Run:
    pip install fastapi uvicorn
    uvicorn app_example:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel

from render_report import render_report

app = FastAPI(title="Dental Report Renderer")


class Prediction(BaseModel):
    # The raw JSON string (or already-parsed object) produced by the VLM.
    prediction: str


@app.post("/report", response_class=HTMLResponse)
def report_html(item: Prediction):
    """Return the report as an HTML page."""
    return render_report(item.prediction, fmt="html")


@app.post("/report.txt", response_class=PlainTextResponse)
def report_text(item: Prediction):
    """Return the report as plain text."""
    return render_report(item.prediction, fmt="text")


# To turn an image into a report end to end, add an endpoint that:
#   1. takes an uploaded image,
#   2. runs your fine-tuned model to get the JSON string,
#   3. calls render_report(json_string, fmt="html") and returns it.
