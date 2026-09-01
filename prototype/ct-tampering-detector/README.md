---
title: CT Tampering Detector
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# CT Tampering Detector

Dockerized Streamlit application for detecting and localizing AI-manipulated
regions in CT scans. See the [repo-level GETTING_STARTED.md](../../GETTING_STARTED.md)
for the full walkthrough (env setup, model weights, dataset access).

## Model weights

Checkpoints are not committed to git — download them from the links in
[`pipeline/models/README.md`](pipeline/models/README.md) and place each file
in its corresponding `pipeline/models/<name>/` subfolder before running.

## Run with Docker

```bash
cd prototype/ct-tampering-detector
docker build -t ct-app .
docker run --gpus all -p 7860:7860 ct-app   # omit --gpus all if you don't have a GPU
```

Then open http://localhost:7860.

## Run locally without Docker

```bash
cd prototype/ct-tampering-detector
pip install -r requirements.txt
streamlit run app.py
```
