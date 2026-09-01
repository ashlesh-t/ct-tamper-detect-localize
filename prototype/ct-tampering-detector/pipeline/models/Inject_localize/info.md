# Injection Localization

Stage 3 of the pipeline. U-Net++ with attention (CBAM + adapter blocks) and a
soft-kNN pixel-refinement head, producing a pixel-level mask of injected
tumor regions. Used by
[`pipeline/localizePipe/Injected.py`](../../localizePipe/Injected.py).

Training code: [`train/UNet_Injection_Expt2.py`](../../../../../train/UNet_Injection_Expt2.py)
(base) and [`train/UNET_injection_fine_tune_v1.3.py`](../../../../../train/UNET_injection_fine_tune_v1.3.py)
(final fine-tune). See [`train/README.md`](../../../../../train/README.md).

Reported result (paper): Dice 0.9201.

Download the checkpoint from the link in [`../README.md`](../README.md).
