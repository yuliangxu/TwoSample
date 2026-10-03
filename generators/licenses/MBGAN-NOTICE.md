# MBGAN source attribution

`generators/mbgan.py` adapts the MBGAN generator/critic architecture, taxonomy-prefix transformation, and alternating Wasserstein training structure from:

* [zhanxw/MB-GAN](https://github.com/zhanxw/MB-GAN), especially `model.py`, `utils.py`, and `mbgan_train_demo.py` (GNU General Public License version 3).
* The recovered research `clean/model.py` and `clean/1_mbgan.ipynb`, which retain the upstream architecture and explicitly credit Keras-contrib. The research notebook supplies the paper-data usage context; its exact historical training membership and resulting checkpoint correspondence remain unverified.
* [keras-team/keras-contrib, examples/improved_wgan.py](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py), credited by the original MBGAN implementation (MIT License; Copyright (c) 2017 Fariz Rahman).

Publication: Rong R, Jiang S, Xu L, et al. *MB-GAN: Microbiome Simulation via Generative Adversarial Network*. GigaScience 10(2), giab005 (2021). [DOI:10.1093/gigascience/giab005](https://doi.org/10.1093/gigascience/giab005).

The 2026 portable adaptation replaces obsolete symbolic Keras training code with explicit TensorFlow gradient tapes, restores the connected gradient penalty, records training provenance, and reads the recovered HDF5 generator without changing its weights. Changes are described in `../MBGAN.md`.

The adapted module is GPL-3.0-only; the complete GPL text is in `MBGAN-GPL-3.0.txt`. The credited MIT license is preserved in `KERAS-CONTRIB-MIT.txt`. This notice concerns the MBGAN adaptation and does not assert a license change for other repository components.
