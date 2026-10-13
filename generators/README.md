# Reproduce the case-study generators

This directory regenerates the final 1,000-row synthetic cohorts in
`data/sample_*.csv`, in report order: **ICFM, DT, D, MBGAN**. Each has the same
123 taxa as the real data. The original real-data preprocessing and all 1,166
training and 292 testing rows are retained.

| Model | Final generation recipe | Seed |
| --- | --- | --- |
| ICFM | Fixed recovered MLP checkpoint; restored torchdyn 1.0.6 numerical path | 2 |
| DT | Stable independent-Beta tree fit to all 1,166 training rows | 1 |
| D | Recovered Newton Dirichlet fit to all 1,166 training rows | 1 |
| MBGAN | Fixed recovered HDF5 generator; original latent draws and prediction batches | 256 |

ICFM and MBGAN reproduce samples from bundled checkpoints. Their original
training membership, RNG seeds and complete fitting settings were not recovered;
the public real test split therefore cannot be asserted to have been held out
from their historical training. The package does not claim to reconstruct
those training runs. DT and D are refitted using only the real training CSV.

## Environments and regeneration

Python 3.9.25 was used for validation. Install the pinned dependencies in two
separate environments because the neural libraries use different NumPy versions:

```sh
python3 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan.txt

.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generators/reproduced --include-real
```

Run from the repository root and choose a new or empty output directory. This
command checks the pinned inputs and generator assets, generates all four
cohorts, compares them with the final CSVs, and writes fitted parametric
checkpoints, logs and `reproduction.json`. It writes `COMPLETE` only when all
requested models pass. `--include-real` copies the unchanged train/test CSVs.
To regenerate one model, add `--models icfm`, `dt`, `d`, or `mbgan`.

CPU sampling with the recorded packages is the reference path. Numerical
reproducibility can depend on hardware, library versions, threads, batching and
floating-point arithmetic. The comparison records exact CSV and numeric matches;
its maximum-absolute-error tolerances are ICFM `1e-6`, DT `1e-8`, D `1e-9`, and
MBGAN `1e-6`. These tolerances check replay, not fit to the real distribution.

To check bundled hashes and valid composition shapes, or validate a generated
bank independently:

```sh
.venv-generators/bin/python -m generators.validate
.venv-generators/bin/python -m generators.validate \
  --data-dir output/generators/reproduced \
  --output output/generators/reproduced_validation
```

Each validation output directory must not already contain `validation.json`.
`manifest.json` pins the six final data CSVs, checkpoints, tree assets and
vendored numerical files. Validation does not compare with earlier experiments.

## Numerical definitions

D clips input entries to `[1e-10, 1]` without renormalization, preserving the
recovered Newton fit. Its final likelihood-score residual must be below `1e-8`.
DT preserves float32 row normalization and `1e-7` child-mass smoothing, fits
positive Beta shape parameters at all 122 internal nodes, and verifies an
interior solution with residual below `1e-8` at every node. Both use the original
NumPy `RandomState` sampling order.

ICFM uses the recorded tree and a float32 inverse sigmoid followed by float64
complements and tree products. MBGAN reconstructs the saved Keras layer chain
and checks every HDF5 weight array. Details and individual sampling commands are
in [ICFM.md](ICFM.md) and [MBGAN.md](MBGAN.md).

The torchdyn subset retains its Apache-2.0 license and source manifest in
`vendor/`. MBGAN retains the GPL-3.0 and credited MIT notices in `licenses/`.
These component notices do not relicense unrelated repository files.
