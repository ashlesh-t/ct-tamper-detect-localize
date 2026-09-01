# Classifier 1 — Real vs. Fake

Stage 1 of the pipeline. DenseNet-121 backbone + attention module + ML
projection head, distinguishing real CT slices from any AI-manipulated
(injected or removed) slice. Used by
[`pipeline/classifierPipe/Real_Fake.py`](../../classifierPipe/Real_Fake.py).

Training code: [`train/RealFakeClissifier.py`](../../../../../train/RealFakeClissifier.py)
(base) and [`train/RealFakeClassifier_FineTune.py`](../../../../../train/RealFakeClassifier_FineTune.py)
(fine-tune). See [`train/README.md`](../../../../../train/README.md).

Reported result (paper): 93.41% accuracy, F1 0.937.

Download the checkpoint from the link in [`../README.md`](../README.md).
