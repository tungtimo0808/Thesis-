# Inference: web app + pretty report

A small FastAPI web interface that turns a panoramic X-ray into a dentist-friendly report,
plus the renderer it uses.

## Files

- `app.py` — the FastAPI web app. Upload an X-ray, the best model (InternVL3-8B fine-tuned
  with QLoRA, adapter `~/pan924_runs/internvl/checkpoint-500`) reads it, and the result is
  shown as a clean report: each region, its teeth (FDI) with colour-coded conditions, the
  per-region notes, and the overall summary. A "draft — review required" banner is always
  shown.
- `render_report.py` — turns the model's JSON report into plain text or HTML. Used by the
  app and usable on its own.
- `requirements.txt` — the extra packages the app needs (on top of the training env).

## Run

On the GPU machine, inside the project venv:

```bash
pip install -r requirements.txt
cd ~/pan924                 # the folder that holds vlm_report_dataset/
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open `http://<server>:8000`. The model loads on the first request (about a minute), then
stays in memory for the next requests.

## Preview the interface without the model

The renderer is independent of the model, so you can see the design with a sample report:

```python
python render_report.py     # prints a demo text report
```

## How it fits together

1. The user uploads a panoramic image.
2. `app.py` runs InternVL3-8B on it and gets the JSON report (the same schema used in
   training and evaluation).
3. The JSON is rendered into the report shown to the user.

The model keeps producing JSON, so it stays measurable against the gold reports; the
pretty report is only the presentation layer.
