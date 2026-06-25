pan924_ver2.zip - report + Faster R-CNN (NOT in pan924_vlm.zip / pan924_gcp_l4.zip)

Contents:
  report/                  report.tex, references.bib, history.txt, figures
  faster_rcnn_fdi/         tooth-number detector: scripts, configs, results json, plots
  faster_rcnn_disease/     condition detector: scripts, configs, results json, plots
  README.md, requirements.txt

Excluded on purpose: model weights (*.pth), original images (*.jpg), and everything
already shipped in pan924_vlm.zip (vlm_report_dataset/) and pan924_gcp_l4.zip (gcp_l4/).

Suggested extract on the VM:
  python3 -c "import zipfile,os; zipfile.ZipFile(os.path.expanduser('~/pan924_ver2.zip')).extractall(os.path.expanduser('~/pan924'))"
  -> gives ~/pan924/report/, ~/pan924/faster_rcnn_fdi/, ~/pan924/faster_rcnn_disease/

To compile the report:  cd ~/pan924/report && pdflatex report.tex (add missing figures first).
