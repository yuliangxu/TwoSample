# Historical local MBGAN variant

This implementation fits a local research variant of MBGAN from random
initialization. It preserves the attributed source in
`vendor/historical_mbgan_model.py` and the GPL/MIT notices in `licenses/`.
It is not a faithful implementation of the published WGAN-GP objective:
the historical graph discards its gradient-penalty branch and applies no weight
clipping. This variant was selected retrospectively in earlier work; its recipe
is fixed before any new run, without test-based variant, seed, or checkpoint search.

The generator uses latent dimension 100, three width-512 Dense/BatchNorm/ReLU
blocks and a softmax over runtime taxa. The critic aggregates taxonomy prefixes,
applies `log(1+1000*x)/log(1001)`, then three width-256 Dense/LeakyReLU/dropout
layers and a scalar output. The effective critic objective is mean(fake score)
minus mean(real score); the generator minimizes minus mean(critic(fake)).

Use the separate pinned environment in `requirements-mbgan.txt`, including
TensorFlow CPU/Keras 2.13.1. The full recipe uses 500,000 generator updates,
five critic updates each, batch 32, RMSprop rate `5e-5`, seed 1, two CPU numerical
threads, oneDNN, and deterministic TensorFlow operations. Sampling uses the final
prescribed update and seed 256; there is no post-sampling thresholding or
renormalization. Diagnostic callbacks use isolated seeded RNG streams rather
than reseeding the training stream from entropy.

```sh
.venv-mbgan/bin/python -m generators.mbgan train \
  --train /absolute/prepared-data/sample_train.csv \
  --taxonomy /absolute/prepared-data/provenance/taxa.txt \
  --output-dir /absolute/new-mbgan
.venv-mbgan/bin/python -m generators.mbgan sample \
  --checkpoint /absolute/new-mbgan/generator.npz \
  --output /absolute/new-mbgan/sample.csv --n 1000 --seed 256
```

Paths must be external and new. The [full workflow](README.md) trains twice,
compares every loss step, model and optimizer arrays, callback outputs, saved RNG
states, and generated sample bytes, and requires complete finite training and
save/reload agreement. These are repeatability checks, not evidence of fidelity
or convergence. Native checkpoints and all outputs are runtime artifacts, not
repository contents. TensorFlow RNG replay positions do not constitute supported
mid-training checkpoint resume.
