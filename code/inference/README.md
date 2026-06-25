# Inference: rendering a pretty report

The fine-tuned models output a JSON report (this is what the evaluation script in
`../dataset_pipeline/eval_report.py` scores). For a user-facing tool, this folder turns
that JSON into a clean, human-readable report.

## Files

- `render_report.py` — the renderer. Turns a JSON report (string or dict) into plain text
  or a self-contained HTML page. No third-party dependencies.
- `app_example.py` — a minimal FastAPI app that serves the rendered report. A starting
  point for the interface.

## Quick use

```python
from render_report import render_report

pretty_text = render_report(model_output_json, fmt="text")
pretty_html = render_report(model_output_json, fmt="html")
```

Try the renderer on its own:

```bash
python render_report.py        # prints a demo report
```

## End-to-end shape (for the FastAPI interface)

1. The user uploads a panoramic image.
2. The fine-tuned VLM is run on the image and returns the JSON report string.
3. `render_report(json_string, fmt="html")` turns that JSON into the report shown to the user.

The model keeps producing JSON (so it stays measurable against the gold reports); the
pretty report is only the presentation layer on top.
