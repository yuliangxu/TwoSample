# Reproduce every generator from training data

This workflow fits **ICFM, DT, D, MBGAN**, in that order, using only the
preserved 1,166 training observations with 123 taxa. It creates 1,000 synthetic
compositions per model. Neural models start from seeded random initialization;
no pretrained model or historical checkpoint is an input. The 292 real test
observations are copied for subsequent analysis and never used for generator
training, early stopping, or model selection.

Both full independent fitting sequences completed. Every model's fitted
parameters and synthetic CSVs matched exactly, with the prescribed neural
training budgets completed in full. See the [reproduction evidence](../data/evidence/reproduction.json)
and [MBGAN trajectory validation](../data/evidence/mbgan_independent_validation.json).

| Model | Frozen fitting recipe | Training seed | Sampling seed |
| --- | --- | ---: | ---: |
| ICFM | Three hidden layers of width 64; 20,000 Adam updates at rate 0.001; batch 256; flow noise 0.0001; final update | 20261003 | 2 |
| DT | Independent positive-parameter Beta maximum-likelihood fits at all 122 tree nodes | deterministic fit | 1 |
| D | Newton Dirichlet maximum-likelihood fit | deterministic fit | 1 |
| MBGAN | 500,000 generator iterations; five critic updates each; RMSprop at rate 0.00005; batch 32; connected gradient penalty; final iteration | 1 | 256 |

[training_config.json](training_config.json) records the full frozen settings.
ICFM uses CPU execution with two numerical threads. Each of the two independent
MBGAN runs uses GPU execution with two host numerical threads and deterministic
TensorFlow operations. Model-specific transformations, architectures, and
numerical definitions are documented in
[ICFM.md](ICFM.md) and [MBGAN.md](MBGAN.md).

## Install and run

Use separate Python environments for the pinned PyTorch and TensorFlow
packages. The reference environments use Python 3.9.25. MBGAN also requires a
working TensorFlow GPU runtime; GPU libraries and device details belong to its
recorded execution environment.

```sh
python3 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan-cuda.txt
```

Before the generation command, apply the [GPU library and determinism
exports](MBGAN.md#gpu-libraries-and-hardware-evidence); the Python orchestrator
does not discover shared-library paths itself. The Slurm launcher performs
that setup automatically. Then run:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generated
```

Run from the repository root and choose an output directory that does not
already exist. The command runs two complete independent fitting and sampling
sequences. With two visible GPUs they run concurrently, one per GPU; with one
GPU they run sequentially. Each sequence starts afresh and uses the same frozen
settings, device type, and seeds. ICFM is trained on CPU and MBGAN on GPU in
both runs. It saves fitted parameters only as intermediate outputs
for sampling and auditing.

Check the regenerated bank
against the published `data/` reference, including all six CSVs, all four fitted
parameter hashes and configurations, and the recorded neural state hashes.
This adds a check against the released results to the two-run equality gate;
exact agreement is scoped to the recorded hardware and software environment:

```sh
python3 scripts/verify_training_reference.py output/generated/data
```

By default, the orchestrator uses the first two identifiers from
`CUDA_VISIBLE_DEVICES`, or a single GPU `0` when that variable is unset.
`--gpu-devices` accepts one or two comma-separated identifiers; when the
scheduler has set `CUDA_VISIBLE_DEVICES`, the explicit identifiers must be a
subset of that allocation. Preserve scheduler-provided GPU visibility.

The output layout is:

- `replicate_1/` and `replicate_2/`: each model's fitted weights or parameters,
  samples, native training metadata, and logs.
- `execution.json`: frozen settings, executable-source hashes, interpreter
  paths, and the empty list of pretrained-model inputs.
- `data/`: unchanged real training/test CSVs, four new generated CSVs,
  `generation_manifest.json`, and compact verification records in `evidence/`.
- `source/`: a snapshot of the generator source files and fitting assets.
- `mbgan_independent_validation.json`: detailed MBGAN trajectory comparison.
- `COMPLETE`: written only after both independent runs pass the checks below.

Use `output/generated/data` as `TWO_SAMPLE_DATA_DIR` for BATTS. The generator
completion marker attests to training and generation only; the analysis writes
its own completion marker after numerical and PDF validation.

## Required reproduction checks

All four models must have identical fitted parameter arrays and byte-identical
synthetic CSVs between the two complete independent runs. The verifier also
checks composition dimensions, finite nonnegative entries, row sums, training
input hashes, source hashes, and completion of the prescribed training budgets.
No tolerance replaces exact equality in this full reproduction gate.

For ICFM, initial parameter hashes and every recorded training loss must agree,
and final parameter hashes must agree after parameters have changed through
training. For MBGAN, the check compares every iteration's recorded losses and
gradient diagnostics, all generator and critic tensors, optimizer-state hashes,
and final RNG state. Wall-clock timings are excluded from trajectory equality.
The MBGAN run also requires finite gradients throughout and an exact save/reload
sampling check.

To revalidate a completed generated bank and its provenance:

```sh
.venv-generators/bin/python -m generators.validate \
  --data-dir output/generated/data \
  --output output/generated_validation
```

The validation output must not already contain `validation.json`.
`manifest.json` binds the preserved real data and fitting assets. The new
`data/generation_manifest.json` within the generation output binds each
training record, independent reproduction record, and generated sample hash.
Exact reproduction is checked within the recorded hardware and software
environment. GPU and CPU arithmetic need not produce the same fitted weights
or samples; matching across device types, GPU models, or library versions is
not assumed. The earlier CPU MBGAN run is preserved separately and cannot
replace one replica of this GPU reproduction pair.

## Parametric definitions and attribution

D clips input entries to `[1e-10, 1]` without renormalization, preserving the
recovered Newton fit. Its final likelihood-score residual must be below `1e-8`.
DT preserves float32 row normalization and `1e-7` child-mass smoothing, fits
positive Beta shape parameters at all 122 internal nodes, and verifies an
interior solution with residual below `1e-8` at every node. Both use the original
NumPy `RandomState` sampling order.

The original preprocessing, real-data split, and taxon order are preserved.
Preprocessing was collection-wide, and the observation split does not establish
participant-level separation. These documented neural training recipes do not
claim to recover unknown historical training trajectories.

The torchdyn subset retains its Apache-2.0 license and source manifest in
`vendor/`. MBGAN retains the GPL-3.0 and credited MIT notices in `licenses/`.
These component notices do not relicense unrelated repository files.
