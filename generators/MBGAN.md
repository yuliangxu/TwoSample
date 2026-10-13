# Fixed MBGAN checkpoint replay

The final MBGAN cohort uses `checkpoints/mbgan_generator.h5`, the recovered
500,000-iteration generator. Its SHA-256 is
`457c13dbba3afefd1c0a7979eb6854118d25806143ac5306bd1543740eaf840f`.
The checkpoint records Keras 2.13.1. The validated replay environment is Python
3.9.25, TensorFlow 2.20.0, Keras 3.10.0, NumPy 2.0.2 and h5py 3.14.0;
install `requirements-mbgan.txt` in a separate environment.

From the repository root:

```sh
python -m generators.mbgan sample \
  --checkpoint generators/checkpoints/mbgan_generator.h5 \
  --n 1000 --seed 256 --batch-size 32 --threads 2 --device cpu \
  --output output/mbgan/sample_mbgan.csv
```

Keras 3 cannot deserialize the original outer Functional wrapper. The loader
reads the saved Sequential layer configuration and assigns all 20 HDF5 weight
arrays, checking each exactly. It preserves the saved activations,
batch-normalization parameters and final 123-dimensional softmax.

Sampling uses NumPy `RandomState` normal latent draws of dimension 100, seed
256, and prediction batches of 32. It applies no thresholding, smoothing,
renormalization or taxon reordering. The final CSV SHA-256 is
`bc4b43fca1d64f9266c064df3e48e7d4979c80043945fe7242ff3c563ce19ad4`.
The output sidecar records checkpoint and output hashes, replay settings,
software versions, verified weight counts and composition checks. Existing
outputs are protected from overwriting. Changing hardware or library versions
may affect floating-point replay.

This reproduces the selected samples from a fixed checkpoint. The checkpoint's
historical training membership and complete fitting provenance are unknown;
this release does not reconstruct that training run.

The module adapts the recovered implementation and
[zhanxw/MB-GAN](https://github.com/zhanxw/MB-GAN). It retains GPL-3.0 and the
credited Keras-contrib MIT notice; see
[licenses/MBGAN-NOTICE.md](licenses/MBGAN-NOTICE.md).
