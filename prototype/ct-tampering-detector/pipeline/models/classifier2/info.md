# Classifier 2 — Injection vs. Removal

Stage 2 of the pipeline. EfficientNet-B2 + DenseNet-121 ensemble,
distinguishing whether a manipulated slice had a tumor **injected** or
**removed**. Used by
[`pipeline/classifierPipe/Inject_Removal.py`](../../classifierPipe/Inject_Removal.py).

Training code: **not available** — see the "training script missing" note in
[`train/README.md`](../../../../../train/README.md). Only the inference code
and this downloadable checkpoint exist currently.

Reported result (paper): macro F1 0.938.

Download the checkpoint from the link in [`../README.md`](../README.md).
