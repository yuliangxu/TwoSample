# ICFM training and checkpoint replay

`python -m generators.icfm` provides separate `train` and `sample` commands.
The bundled recovered checkpoint reproduces the selected regenerated samples.
Training creates a new model using an explicit reproducible recipe; it does not
claim to recover the historical checkpoint's unknown training settings or data
membership.

The architecture is the recovered time-dependent MLP: 122 log-tree coordinates
plus one time input, three hidden layers of width 64 with SELU activations, and
122 output velocities. The three tree CSVs under `generators/assets/tree/` fix
the taxon ordering and inverse transformation.

## Reproduce the selected synthetic bank

From the repository root:

```bash
python -m generators.icfm sample \
  --checkpoint generators/checkpoints/icfm_recovered.pt \
  --seed 2 --n 1000 --device cpu --threads 2 \
  --atol 1e-4 --rtol 1e-4 \
  --output outputs/icfm_recovered_seed2.npy
```

The tested CPU environment produces an NPY SHA-256 of
`c1043b018c7bb554577a1fc21cddc590c103c23bd66c8b70a351c12744e1418d`.
Use an output ending in `.csv` for a headerless CSV instead. The adjacent
`*.metadata.json` records the checkpoint and tree hashes, random seed, solver,
precision, device, software versions, sample count, and output checksum.

Sampling uses the source-preserving numerical subset of torchdyn 1.0.6 shipped
under `generators/vendor/`; it does not require torchdyn, PyTorch Lightning,
torchdiffeq, or attrs to be installed. Upstream source hashes, extracted
definition locations, modifications, and the Apache 2.0 license are included.
The current torchdyn version is documented; the historical notebook did not
record its original package version.

The inverse intentionally computes the sigmoid in float32, stores the result in
float64, and computes the complement and tree products in float64. Changing
this arithmetic, the ODE solver, device, or dependency versions can change
sample values. CPU and CUDA random draws differ even with the same seed.

## Train a new model using public training data

```bash
python -m generators.icfm train \
  --train data/sample_train.csv \
  --steps 20000 --batch-size 256 --lr 1e-3 --sigma 1e-4 \
  --seed 20261003 --device cpu --threads 2 \
  --zero-mass 1e-8 --clip-logratio 13.815510557964274 \
  --output outputs/icfm_new_training.pt

python -m generators.icfm sample \
  --checkpoint outputs/icfm_new_training.pt \
  --seed 2 --n 1000 --device cpu \
  --output outputs/icfm_new_training_seed2.npy
```

These are **new training defaults**, not inferred historical settings. They
make the recovered model family trainable from the public 1,166-row training
matrix. The command loads only its supplied training CSV; it never loads the
public test set. It saves the final optimization iterate without validation,
early stopping, or checkpoint selection. Full 20,000-step training was not run
as part of the selected-sample release.

Preprocessing normalizes each composition in float64, sums child-node masses,
replaces zero child masses with `1e-8`, and clips log ratios to
`[-log(1e6), log(1e6)]`, before conversion to float32. This transformation was
checked against archived transformed data. Its agreement does not establish
which observations were used to train the recovered checkpoint.

Each training step independently samples training rows with replacement,
standard-normal `z0`, time `t ~ Uniform(0,1)`, and standard-normal `epsilon`.
It uses

```text
zt = (1 - t) * z0 + t * z1 + sigma * epsilon
target_velocity = z1 - z0
loss = mean((network(zt, t) - target_velocity)^2)
```

Adam uses learning rate `1e-3`, betas `(0.9, 0.999)`, epsilon `1e-8`, and no
weight decay. Python, NumPy, and Torch RNGs are seeded explicitly, and Torch
deterministic algorithms are enabled. The saved checkpoint and adjacent JSON
record the input hash, transformation settings, optimizer, number of updates,
batch size, equivalent passes through the data, complete loss trace, initial
and final tensor fingerprints, device, and dependency versions. Existing output
files and the bundled recovered checkpoint are protected from overwriting.
Sampling a newly trained checkpoint verifies that the supplied tree CSV hashes
match the tree recorded during training; a mismatched tree is rejected before
generation.

## Validation performed

- Replayed 1,000 recovered-checkpoint samples and matched the selected NPY
  checksum exactly.
- Ran two independent CPU training processes for eight updates, batch size 32,
  seed 20261003, using all 1,166 public training rows as the sampling bank.
  Both changed parameters and produced identical tensors and loss traces.
- Loaded a newly trained checkpoint through the sampling command and generated
  a separate 17-row valid composition bank.
- Verified that changing a training-tree file causes sampling of the new
  checkpoint to fail before generation.

The short runs verify executable training, deterministic repetition, and
checkpoint compatibility; they do not establish convergence or model quality.
Their temporary weights are not the bundled selected checkpoint. See
`ICFM_validation.json` for the recorded checks and environment.
