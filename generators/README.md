# Generative models: fitting, checkpoint replay, and selected samples

The selected regenerated matrices are in
[`data/regenerated_20261003/`](../data/regenerated_20261003/). Each contains
1,000 compositions over the same 123 taxa as the real data. The release covers
Dirichlet, stabilized Dirichlet tree, ICFM, and MBGAN.

Two reproducibility tasks are distinguished:

1. **Regenerate the selected samples.** Dirichlet and DT are fitted from the
   public training CSV. ICFM and MBGAN are sampled from bundled recovered
   checkpoints. This is the validated route to the selected matrices.
2. **Train new neural models.** Runnable ICFM and MBGAN training commands use
   the public training CSV and record all settings. Their explicit recipes
   are new reconstructions: the original neural fit seeds, data membership,
   and complete settings were not recovered. They do not recreate the bundled
   historical weights or claim to reproduce the selected neural samples.

| Model | Fitting code | Selected-sample route | Training provenance |
| --- | --- | --- | --- |
| Dirichlet | [`dirichlet.py`](dirichlet.py) | Train-only Newton MLE, seed-1 Dirichlet draws | All 1,166 public training rows |
| DT | [`dirichlet_tree.py`](dirichlet_tree.py) | Train-only stable Beta MLE at 122 nodes, seed-1 nodewise draws | All 1,166 public training rows |
| ICFM | [`icfm.py`](icfm.py) | Recovered MLP checkpoint, restored torchdyn sampler, seed 2 | Historical checkpoint membership unknown; new training command uses supplied training CSV only |
| MBGAN | [`mbgan.py`](mbgan.py) | Recovered HDF5 generator, seed 256 | Historical checkpoint membership unknown; new training command uses supplied training CSV only |

The released training commands use their supplied training data; evaluation
commands also read the 292 public real test observations. The
original paper matrices in `data/sample_*.csv` remain paired with their
existing BATTS posterior results. The updated synthetic matrices require new
downstream fits; this release does not refit BATTS.

## Install the tested environments

Validation used Python 3.9.25. The pinned packages support Python 3.9–3.12.
Use separate environments for the NumPy 1.26 parametric/PyTorch path and the
NumPy 2.0 TensorFlow path:

```sh
python3 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan.txt
```

The ICFM sampler includes the required torchdyn 1.0.6 numerical definitions;
no torchdyn, torchcfm, Lightning, or event-callback dependencies are needed.
See [`ICFM.md`](ICFM.md) for the exact numerical path and upstream attribution.
MBGAN loads the original Keras-2 HDF5 weights into the equivalent Keras-3
layer chain and verifies every weight array.

## Regenerate all four selected samples

Run from the repository root:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generators/reproduced --include-real
```

The output directory must be new or empty. The command:

- fits Dirichlet and stabilized DT using only `data/sample_train.csv`;
- samples the bundled neural checkpoints with the recorded CPU settings;
- writes four synthetic CSVs, fitted parametric checkpoints, per-model
  metadata and logs, and an aggregate `reproduction.json`;
- compares each result with the selected CSV using documented numerical
  tolerances, recording exact matches as well;
- writes `COMPLETE` only after every requested model passes;
- with `--include-real`, copies the unchanged real train/test CSVs alongside
  the synthetic files for use in a new downstream analysis.

To run one family, add `--models dt`, `--models icfm`, or another model key.
CPU sampling is the reference path. Changing devices, package versions,
batching, or floating-point implementations can change generated rows even
with the same integer seed.

Check bundled file integrity and compare with real data independently of
model dependencies:

```sh
.venv-generators/bin/python -m generators.validate --check-bundled-hashes
.venv-generators/bin/python -m generators.validate \
  --data-dir output/generators/reproduced \
  --output output/generators/reproduced_validation
.venv-generators/bin/python -m generators.compare_icfm \
  --bootstrap 1000 --output output/generators/icfm_comparison
```

The evaluator reports unbiased energy statistics, mean-abundance and
prevalence errors, diversity, dominance, and richness on both real splits.
See the [release report](../results/generators_20261003/REPORT.md) for the
selected-model comparisons and verification evidence.

## Fit and sample the parametric models separately

```sh
.venv-generators/bin/python -m generators.dirichlet fit-sample \
  --train data/sample_train.csv \
  --checkpoint output/dirichlet/model.npz \
  --output output/dirichlet/sample_d.csv --seed 1 \
  --report output/dirichlet/fit.json

.venv-generators/bin/python -m generators.dirichlet_tree fit-sample \
  --train data/sample_train.csv \
  --checkpoint output/dt/model.npz \
  --output output/dt/sample_dt.csv --seed 1 \
  --report output/dt/fit.json
```

Both modules also expose `fit` and `sample` subcommands. Fitted parameters
are bundled in `checkpoints/` for convenient sampling, but the combined
regeneration command refits them from the real training data.

Dirichlet uses the recovered damped Newton iteration and the legacy NumPy
rowwise Dirichlet sampler. DT models independent Beta splits on a fixed
binary tree. It preserves the recovered float32/`1e-7` preprocessing and
sampling order, while solving the Beta likelihood equations in positive log
parameters with an analytic Jacobian. Every node must have a finite interior
solution with maximum score residual below `1e-8`; invalid fits fail.

The archived ICFM training transformation is different from DT's: zero child
masses become `1e-8` and log-ratios are clipped to `±log(1e6)`. Both methods
are explicit in [`tree.py`](tree.py). The tree metadata and public taxonomy
ordering are checked when loaded.

## Train a new ICFM model

```sh
.venv-generators/bin/python -m generators.icfm train \
  --train data/sample_train.csv --output output/icfm_new/model.pt \
  --seed 20261003 --steps 20000 --batch-size 256 --lr 0.001 --sigma 0.0001
.venv-generators/bin/python -m generators.icfm sample \
  --checkpoint output/icfm_new/model.pt \
  --output output/icfm_new/sample_icfm.csv --n 1000 --seed 2
```

This keeps the recovered `123 → 64 → 64 → 64 → 122` SELU MLP. With independent
Gaussian `z0`, a sampled training log-ratio vector `z1`, uniform time `t`,
and Gaussian `epsilon`, it minimizes squared velocity error at
`zt=(1-t)*z0+t*z1+sigma*epsilon`, with target `z1-z0`. The final update is saved;
there is no test-based model selection. Exact seeds, input hashes,
preprocessing, optimizer settings, losses, and environment are retained.

These defaults are an explicit new recipe, not recovered historical fitting
settings. Deterministic short runs validate training and checkpoint loading;
a full new fit and its scientific adequacy have not been evaluated here.

## Train a new MBGAN model

```sh
.venv-mbgan/bin/python -m generators.mbgan train \
  --train data/sample_train.csv --taxonomy data/provenance/taxa.txt \
  --output-dir output/mbgan_new --iterations 500000 --seed 1
.venv-mbgan/bin/python -m generators.mbgan sample \
  --checkpoint output/mbgan_new/generator.npz \
  --output output/mbgan_new/sample_mbgan.csv --n 1000 --seed 256
```

This adapts the recovered MBGAN architecture and taxonomy-aware critic to
modern TensorFlow training. The explicit new recipe uses a connected
Wasserstein gradient penalty; the recovered legacy training graph did not
connect that penalty correctly. This is a documented training repair, not
a claim to reproduce historical weights. The long iteration budget is shown
explicitly; only short training checks were run for this code release.
See [`MBGAN.md`](MBGAN.md) for architecture, settings, validation, and licensing.

## Source attribution and licenses

The vendored torchdyn definitions retain their Apache-2.0 license and source
hashes. The MBGAN adaptation retains GPL-3.0 and the credited MIT notices;
see `licenses/` and the component documentation. The generator data use the
case-study taxonomy and real-data provenance already documented in
[`data/`](../data/README.md). No blanket relicensing of the existing repository
is implied by these third-party notices.
