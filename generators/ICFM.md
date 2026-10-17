# Independent conditional flow matching

The model starts from seeded random initialization and reads only an explicitly
supplied external training CSV and taxonomic tree. It does not use pretrained
weights, test rows, or checkpoint selection.

The prescribed transform normalizes compositions in float64, computes left/right
subtree masses, replaces exactly zero masses by `1e-8`, takes log mass ratios,
and clips them to `[-log(1e6), log(1e6)]`. The dimension is one less than the
runtime number of taxa. A time-dependent vector field has three width-64 SELU
hidden layers and a linear output.

At each of 20,000 Adam updates (batch 256, rate `1e-3`, seed 20261003), sample
training `z1`, independent standard-normal `z0` and `epsilon`, and uniform time
`t`. Regress velocity `z1-z0` at `(1-t)*z0+t*z1+1e-4*epsilon` using mean squared
error. Keep the final update; no validation selection or early stopping occurs.
Execution uses CPU float32, deterministic PyTorch operations, and two threads.

Generation starts at 1,000 standard-normal points with seed 2 and integrates the
preserved torchdyn 1.0.6 Dormand-Prince path over `[0,1]`, with absolute/relative
tolerances `1e-4`. The tree inverse uses float32 sigmoid and float64 complements
and products. It does not restore structural zeros or threshold generated values.

```sh
python3 -m generators.icfm train \
  --train /absolute/prepared-data/sample_train.csv \
  --tree /absolute/prepared-data/tree --output /absolute/new-icfm/model.pt
python3 -m generators.icfm sample \
  --train /absolute/prepared-data/sample_train.csv \
  --tree /absolute/prepared-data/tree \
  --checkpoint /absolute/new-icfm/model.pt \
  --output /absolute/new-icfm/sample.csv
```

The sampler checks the completed training metadata, input/tree/source hashes,
initialization, final parameter state, and numerical environment. Two independent
fits are compared by the full [generator workflow](README.md). Standalone short
training calls are implementation checks and cannot satisfy that full-budget gate.
Checkpoint files are generated locally and must not be committed.
