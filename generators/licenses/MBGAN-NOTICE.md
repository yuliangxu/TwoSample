# MBGAN source attribution

`generators/mbgan.py` adapts the MBGAN generator architecture and checkpoint-loading implementation from:

* [zhanxw/MB-GAN](https://github.com/zhanxw/MB-GAN), especially `model.py`, `utils.py`, and `mbgan_train_demo.py` (GNU General Public License version 3).
* The recovered research `clean/model.py` and `clean/1_mbgan.ipynb`, which retain the upstream architecture and explicitly credit Keras-contrib. The research notebook supplies the paper-data usage context; its exact historical training membership and resulting checkpoint correspondence remain unverified.
* [keras-team/keras-contrib, examples/improved_wgan.py](https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py), credited by the original MBGAN implementation (MIT License; Copyright (c) 2017 Fariz Rahman).

Publication: Rong R, Jiang S, Xu L, et al. *MB-GAN: Microbiome Simulation via Generative Adversarial Network*. GigaScience 10(2), giab005 (2021). [DOI:10.1093/gigascience/giab005](https://doi.org/10.1093/gigascience/giab005).

The portable case-study adaptation reads the recovered HDF5 generator through the saved Sequential layer configuration without changing its weights. New-training and critic code are excluded from this final-result release. The replay is described in `../MBGAN.md`.

The adapted module is GPL-3.0-only; the complete GPL text is in `MBGAN-GPL-3.0.txt`. The credited MIT license is preserved in `KERAS-CONTRIB-MIT.txt`. This notice concerns the MBGAN adaptation and does not assert a license change for other repository components.
