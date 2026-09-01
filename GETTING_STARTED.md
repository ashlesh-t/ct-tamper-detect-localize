# Getting Started

A practical walkthrough for running this project locally. For the project
overview, paper, and architecture, see [`README.md`](README.md).

## 1. Clone and set up a Python environment

Python 3.10 is required (see `.envrc` / `setup.sh`).

```bash
git clone <this-repo>
cd Detect-and-Locate-Tampered-Medical-Images
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Training scripts (in `train/`) need a few extra packages — use
`requirements-train.txt` instead:

```bash
pip install -r requirements-train.txt
```

## 2. Download model weights

Trained model checkpoints are **not committed to git**. Download all four
from the Google Drive links in
[`prototype/ct-tampering-detector/pipeline/models/README.md`](prototype/ct-tampering-detector/pipeline/models/README.md)
and place each `.pth` file in its corresponding `pipeline/models/<name>/`
subfolder. That README is the single source of truth for weight downloads —
every other doc in this repo just links to it.

## 3. Run the Streamlit app

```bash
cd prototype/ct-tampering-detector
docker build -t ct-app .
docker run --gpus all -p 8501:8501 ct-app   # omit --gpus all if you don't have a GPU
```

Or run it directly without Docker:

```bash
cd prototype/ct-tampering-detector
pip install -r requirements.txt
streamlit run app.py
```

## 4. Requesting dataset access

The original CT dataset isn't publicly redistributed — see
[`data/README.md`](data/README.md) for what's available (derived coordinate
CSVs) and how to request the rest.

## 5. Running the test suite

Fast, GPU-free unit tests covering the pure preprocessing/forensic-filter
functions and config path resolution:

```bash
pytest tests/unit/
```

Manual evaluation/inference scripts (need a GPU + downloaded checkpoints, not
part of CI) live in `scripts/eval/`.

## 6. Where the training code lives

See [`train/README.md`](train/README.md) for the full pipeline lineage
(base → fine-tune scripts per stage) and a note on the one stage whose
original training script is currently missing.

## Questions

Dataset access, missing training code, or anything else:
ashleshat5@gmail.com
