# MBGAN source attribution

`generators/vendor/historical_mbgan_model.py` preserves the historical local
model source byte for byte; `generators/mbgan.py` wraps it for new training and
verification. This was the model implementation used for the newly trained
IBDMDB case study. Its architecture and training code
derive from:

- [zhanxw/MB-GAN](https://github.com/zhanxw/MB-GAN), especially `model.py`,
  `utils.py`, and `mbgan_train_demo.py` (GNU General Public License version 3).
- The recovered research `clean/model.py` and `clean/1_mbgan.ipynb`, frozen for
  the IBDMDB training run under `historical_code/source/`. The model source has
  SHA-256 `9efc755e7fdcc18310c5fe4405319134f5d6e25ea50afb0a7b776b6971968e18`.
- [keras-team/keras-contrib, examples/improved_wgan.py](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py),
  credited by the original MBGAN implementation (MIT License; Copyright (c)
  2017 Fariz Rahman).

Publication: Rong R, Jiang S, Xu L, et al. *MB-GAN: Microbiome Simulation via
Generative Adversarial Network*. GigaScience 10(2), giab005 (2021).
[DOI:10.1093/gigascience/giab005](https://doi.org/10.1093/gigascience/giab005).

This release restores the historical local Keras 2.13.1 training graphs,
including their lack of an effective gradient penalty or weight clipping. It
must not be described as the published WGAN-GP implementation. The wrapper
adds explicit provenance and independent-run verification and replaces the
historical callback's entropy reseeding with isolated seeded diagnostics.
It fits new weights from training observations; historical HDF5 checkpoints
are not inputs. See [MBGAN methods and provenance](../MBGAN.md).

The adapted MBGAN component is GPL-3.0-only; the full GPL text is in
`MBGAN-GPL-3.0.txt`. The credited MIT license is preserved in
`KERAS-CONTRIB-MIT.txt`. These notices concern the MBGAN component and do not
relicense unrelated repository files.
