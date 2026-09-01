# Training Scripts

These scripts were written for Kaggle/Colab notebooks (hardcoded `/kaggle/...` or
`/content/drive/...` paths, `drive.mount(...)` calls) and are kept here for
reproducibility and reference alongside the paper. They are not wired into the
Streamlit app in `prototype/` — that app only runs inference against the
downloadable checkpoints listed in
[`pipeline/models/README.md`](../prototype/ct-tampering-detector/pipeline/models/README.md).

## Pipelines

### Injection localization (U-Net++)
1. `UNet_Injection_Expt2.py` — trains the base U-Net++ model
   (`resnet34` encoder, soft-KNN pixel head) from ImageNet init.
2. `UNET_injection_fine_tune_v1.3.py` — loads that checkpoint
   (`prev_experiment = "unetpp_v8_softknn_injection_only_v2"`) and fine-tunes it
   further: `resnet50` encoder, CBAM + adapter enhancement blocks, deep
   supervision, curriculum sampling. This is the final version used for the
   paper's reported injection-localization results (Dice 0.9201).

### Removal localization (U-Net)
- `CT_Removal.py` — trains the streamlined U-Net used for removal
  localization (Dice 0.7369 in the paper).

### Real vs. Fake classification (Stage 1)
1. `RealFakeClissifier.py` — trains the DenseNet-121 + attention + ML-head
   real-vs-fake classifier (balanced sampling, K-fold evaluation). This is the
   final version referenced by `pipeline/classifierPipe/Real_Fake.py`.
2. `RealFakeClassifier_FineTune.py` — fine-tunes the checkpoint produced by
   `RealFakeClissifier.py` (mixup, TTA, threshold tuning). Run
   `RealFakeClissifier.py` first so its checkpoint exists, then run this.

### Injection vs. Removal classification (Stage 2) — training script missing
The paper's Stage 2 (EfficientNet-B2 + DenseNet-121 ensemble for
injection-vs-removal classification) has inference code in
[`pipeline/classifierPipe/Inject_Removal.py`](../prototype/ct-tampering-detector/pipeline/classifierPipe/Inject_Removal.py)
and a downloadable checkpoint
([`pipeline/models/classifier2/`](../prototype/ct-tampering-detector/pipeline/models/classifier2/)),
but **the original training script was not preserved** and could not be
located. This README will be updated if/when it's recovered.

## `archive/`
Superseded/earlier iterations kept for reference, not used to produce the
paper's reported results:
- `UNET_injection_fine_tune_v1.0.py`, `v1.1.py` — earlier iterations of the
  injection fine-tuning script, superseded by `v1.3`.
- `RealFake_Classifier_v1_baseline.py` — an earlier, simpler draft of the
  real-vs-fake classifier (no attention module), superseded by
  `RealFakeClissifier.py`.

## Questions
Training code, data access, or the missing Stage 2 script:
ashleshat5@gmail.com
