# Removal Localization

Stage 4 of the pipeline. A streamlined U-Net producing a pixel-level mask of
regions where a tumor was removed/inpainted. Used by
[`pipeline/localizePipe/Removed.py`](../../localizePipe/Removed.py).

Training code: [`train/CT_Removal.py`](../../../../../train/CT_Removal.py).
See [`train/README.md`](../../../../../train/README.md).

Reported result (paper): Dice 0.7369.

Download the checkpoint from the link in [`../README.md`](../README.md).
