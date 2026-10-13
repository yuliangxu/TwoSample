# ICFM from training data

This implementation trains an independent conditional flow matching model from
random initialization using only `data/sample_train.csv`. No pretrained or
historical weights are accepted. The recipe is the earlier reproducible training
implementation, with explicit numerical settings and provenance checks; it is
not a reconstruction of undocumented historical fitting settings.

## Fixed protocol

- Training input: all 1,166 rows and 123 taxa of the preserved pooled training
  matrix, in the original order. The test CSV is never opened by this module.
- Normalize each composition in float64. Use the fixed binary taxonomic tree to
  compute the left/right subtree masses at its 122 internal nodes. Replace each
  exactly zero mass by `1e-8`, take the log mass ratio, clip to
  `[-log(1e6), log(1e6)]`, then cast the 122 coordinates to float32.
- Vector field: concatenate time to the 122 coordinates; three hidden layers of
  width 64 with SELU activation, and a linear 122-dimensional output.
- Initialization/training seed: `20261003`; 20,000 Adam updates; batch size 256;
  learning rate `1e-3`; betas `(0.9, 0.999)`; epsilon `1e-8`; no weight decay,
  gradient clipping, scheduler, EMA, validation selection, or early stopping.
- At each update, sample training rows uniformly with replacement as `z1`, draw
  independent standard-normal `z0` and `epsilon`, and draw `t` uniformly on
  `[0, 1)`. Regress the velocity on `z1-z0` at
  `zt = (1-t)*z0 + t*z1 + 1e-4*epsilon` using mean squared error over all batch
  coordinates. This is independent CFM with constant conditional noise.
- Retain the parameters from the final update. Use CPU float32, two intra-op
  threads, one inter-op thread, deterministic PyTorch algorithms, and explicit
  non-fused/non-foreach Adam updates.
- Generation seed: `2`; 1,000 standard-normal starting points; integrate from
  zero to one with the preserved torchdyn 1.0.6 Dormand-Prince path at absolute
  and relative tolerances `1e-4`. Apply the preserved tree inverse: float32
  sigmoid, then float64 complements and tree products.

The constant-noise conditional path has standard deviation `1e-4` at both
endpoints; generation starts from a standard normal as in the declared recipe.
The forward transformation handles existing zeros explicitly. The inverse does
not restore the observed zero pattern or perform post-generation thresholding.

## Reproduction

From the repository root, in the pinned generator environment:

```bash
python3 -m generators.icfm train --train data/sample_train.csv \
  --output output/generators/icfm/model.pt
python3 -m generators.icfm sample --train data/sample_train.csv \
  --checkpoint output/generators/icfm/model.pt \
  --output output/generators/icfm/sample_icfm.csv
```

The training command has no input-checkpoint argument and always starts afresh.
The output checkpoint is an intermediate result of this training command, not
an external input to the full reproduction. Both commands refuse to overwrite
existing outputs. Repeat in another output directory to check the final tensor
hash, every training loss, and generated CSV bytes. Checkpoint archive bytes can
differ because runtime metadata and archive filenames differ.

The checkpoint contains initial and final parameters, complete training losses,
the seed and all settings, software versions, and SHA-256 fingerprints of the
training CSV, tree assets, transformed training array, and all executable ICFM,
tree, and numerical-solver sources. A sidecar records the checkpoint file hash.
Sampling requires an explicit checkpoint produced by this implementation and
verifies the sidecar, final and initial parameter hashes, seeded initialization,
training-input hash, tree hashes, source hashes, and software versions. Raw
historical state dictionaries and checkpoints without completed training
provenance are rejected. These are integrity and consistency checks, not signed
attestations; independent retraining establishes empirical reproducibility.

Exact numerical reproducibility is scoped to the pinned CPU software and
floating-point environment. Progress and elapsed/estimated remaining seconds
are logged every 1,000 updates and at completion. Runtime metadata are
descriptive and do not enter optimization or checkpoint selection.
