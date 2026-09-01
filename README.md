# Unveiling AI-Manipulated Medical Images and Localization of Tampered Areas

## Overview
This project detects AI-manipulated CT scans and localizes the tampered
regions. Generative models can insert synthetic tumors into, or erase real
lesions from, medical CT scans — this pipeline flags such manipulations and
produces a pixel-level mask of where they occurred.

## Paper
This repo is the code companion to:

> **Unveiling AI-Manipulated Medical Images: Detecting and Localizing Tampered Areas**
> T. Ashlesha, Vignesh, S. Varun, Vats Shubhangi, Narayan Surabhi — PES University, Bengaluru
> [SciTePress, 2026](https://www.scitepress.org/publishedPapers/2026/146307/pdf/index.html)

Reported results: 93.41% real/fake detection accuracy (F1 0.937), 0.938
macro F1 for injection-vs-removal classification, 0.9201 Dice for injection
localization, 0.7369 Dice for removal localization.

## How it works
A four-stage pipeline:
1. **Real vs. fake** — DenseNet-121 (spatial) + frequency-domain features.
2. **Injection vs. removal** — EfficientNet-B2 + DenseNet-121 ensemble.
3. **Injection localization** — U-Net++ with attention + soft-kNN refinement.
4. **Removal localization** — a streamlined U-Net.

## Project structure
```
prototype/ct-tampering-detector/   The runnable app (Streamlit + inference pipeline)
train/                              Training scripts for each pipeline stage
notebooks/                          Exploratory notebooks and saved experiment results
scripts/eval/                       Manual evaluation/inference driver scripts (not CI tests)
tests/unit/                         Automated pytest unit tests (GPU-free)
Data_coordinates/, data/, utils/    Dataset-prep scripts (private dataset — see data/README.md)
hld.html, lld.html                  High/low-level design docs
```

## Quickstart
See [`GETTING_STARTED.md`](GETTING_STARTED.md) for environment setup,
downloading model weights, running the app, requesting dataset access, and
running tests.

## Technologies Used
- Python, PyTorch
- OpenCV, NumPy, Matplotlib
- Streamlit (app), Docker (deployment)

## Authors
- Ashlesha T - [GitHub Profile](https://github.com/ASHLESHA05)
- Varun S - [GitHub Profile](https://github.com/varuns2903)
- Vignesh - [GitHub Profile](https://github.com/Vignesh3613)
- Shubhangi Vats - [GitHub Profile](https://github.com)

## Contributing
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Questions / Access
Dataset access requests, questions about missing training code (see
[`train/README.md`](train/README.md)), or anything else:
**ashleshat5@gmail.com**

## Acknowledgments
- Thanks to all contributors and open-source libraries used in this project.
