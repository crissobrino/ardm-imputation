# Order-Agnostic Autoregressive Diffusion Models for Imputation

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementation of the Order-Agnostic Autoregressive Diffusion Model (OA-ARDM) from
[Hoogeboom et al., ICLR 2022](https://openreview.net/forum?id=Lm8T39vLDTE), applied to
imputation on images and tabular data.

ARDM trains a model to predict any token given any subset of the others, using a random
generation order at each training step rather than a fixed left-to-right one. Since the model
already conditions on arbitrary subsets, it can impute missing values directly at inference
time with no changes to the architecture or training procedure.

Started as coursework for an MSc module on probabilistic and generative ML, kept here as a
portfolio version.

## What's here

- A bidirectional Transformer trained with the ARDM objective on binarized MNIST
- Unconditional generation, sequential and parallel block sampling
- Imputation from partial observations at 90%, 50%, 10% visibility
- Uncertainty estimates via pixel-wise entropy over repeated completions
- A small CNN as a baseline, to see how much the global receptive field matters
- The same setup applied to a tabular dataset (UCI Mushroom, 22 categorical features)

## Results

Train and validation loss over training, both well below the ln(2) random baseline:

![Training curves](assets/training_curves.png)

Unconditional samples: sequential sampling (top row) vs. parallel block sampling (bottom
row), which is much faster but noticeably lower quality:

![Unconditional generation](assets/unconditional_generation.png)

Imputation at different observation rates. Reconstructions hold up well down to 50%
visibility and degrade gracefully at 10%:

![Imputation reconstruction](assets/imputation_reconstruction.png)

Pixel-wise entropy over 50 completions per image, showing where the model is confident vs.
uncertain about the missing pixels:

![Uncertainty entropy maps](assets/uncertainty_entropy.png)

Same imputation task with the Transformer and the small CNN side by side. The CNN keeps up
at high visibility but falls apart at 10%, where the Transformer's global attention still
finds useful context:

![Transformer vs CNN](assets/transformer_vs_cnn.png)

Imputation on the Mushroom dataset, accuracy stays above the random baseline even with only
10% of features observed:

![Tabular imputation](assets/tabular_imputation.png)

## Repository structure

```
.
├── notebooks/
│   └── ardm_imputation.ipynb   # implementation, training, analysis
├── assets/                     # figures used above
├── requirements.txt
└── LICENSE
```

## Running the notebook

Developed and trained on Google Colab with checkpoints saved to Google Drive, so the
save/load paths near the top of each section assume that setup. To run elsewhere:

1. `pip install -r requirements.txt`
2. Drop the `google.colab.drive` mount cell and point `SAVE_PATH` at a local directory.
3. Run top to bottom. MNIST and the Mushroom dataset are both fetched automatically
   (`torchvision.datasets`, `sklearn.datasets.fetch_openml`), no manual data setup needed.
   Checkpoints aren't included here, so training needs to be re-run to reproduce the
   evaluation cells.

## References

- Hoogeboom, E., Gritsenko, A. A., Bastings, J., Poole, B., van den Berg, R., & Salimans, T.
  (2022). [Autoregressive Diffusion Models](https://openreview.net/forum?id=Lm8T39vLDTE). ICLR.
- Transformer blocks build on Andrej Karpathy's [minGPT](https://github.com/karpathy/minGPT),
  wrapped following
  [this tutorial](https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/).

## License

MIT, see [LICENSE](LICENSE).
