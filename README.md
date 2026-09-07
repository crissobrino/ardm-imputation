# Order-Agnostic Autoregressive Diffusion Models for Imputation

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementation of OA-ARDM ([Hoogeboom et al., ICLR 2022](https://openreview.net/forum?id=Lm8T39vLDTE))
for imputation on images and tabular data. The model is trained to predict any token given
any subset of the others, so it can fill in missing values at inference time with no changes
to the architecture.

- Bidirectional Transformer trained on binarized MNIST, unconditional generation via
  sequential and parallel block sampling
- Imputation at 90/50/10% visibility, with uncertainty via pixel-wise entropy
- Small CNN baseline for comparison
- Same setup applied to a tabular dataset (UCI Mushroom)

## Results

![Training curves](assets/training_curves.png)

![Unconditional generation](assets/unconditional_generation.png)

![Imputation reconstruction](assets/imputation_reconstruction.png)

![Uncertainty entropy maps](assets/uncertainty_entropy.png)

![Transformer vs CNN](assets/transformer_vs_cnn.png)

![Tabular imputation](assets/tabular_imputation.png)

## Structure

```
notebooks/ardm_imputation.ipynb   # implementation, training, analysis
assets/                           # figures above
requirements.txt
```

## Running

Built for Google Colab (Drive-mounted checkpoints), so `SAVE_PATH` and the drive-mount cell
need adjusting to run locally. Otherwise: `pip install -r requirements.txt` and run top to
bottom; MNIST and Mushroom both download automatically.

## References

- Hoogeboom et al., [Autoregressive Diffusion Models](https://openreview.net/forum?id=Lm8T39vLDTE), ICLR 2022
- Transformer blocks based on [minGPT](https://github.com/karpathy/minGPT), wrapped following
  [this tutorial](https://buomsoo-kim.github.io/attention/2020/04/21/Attention-mechanism-19.md/)

MIT license, see [LICENSE](LICENSE).
