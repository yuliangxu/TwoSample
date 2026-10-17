# Train and verify all four generators

This repository contains executable source and method settings. Real matrices,
cohort provenance, taxonomic tree assets, generated samples, model checkpoints,
logs, figures, and completion records belong to external runtime directories.
Nothing from a completed analysis is needed inside this checkout.

Prepare an external data directory with the preprocessing commands in
[the root documentation](../PREPROCESSING.md). It must contain
`sample_train.csv`, `sample_test.csv`, `provenance/cohort_manifest.json`,
`provenance/taxa.txt`, identity/provenance files, and
`tree/{childNode.csv,treeTaxa.csv,outTaxa.csv}`. The runtime cohort manifest binds
all these files by SHA-256; tree output labels must match matrix column order.

Install separate environments using `requirements.txt` (PyTorch) and
`requirements-mbgan.txt` (historical TensorFlow CPU/Keras). The recorded numerical
recipe uses Python 3.9, two CPU numerical threads, seeded random initialization,
and final prescribed updates. The full workflow is:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --data-dir /absolute/prepared-data \
  --tensorflow-python "$PWD/.venv-mbgan/bin/python" \
  --output /absolute/new-generation
.venv-generators/bin/python -m generators.validate \
  --data-dir /absolute/new-generation/data
```

The output directory must be new and outside this checkout. Two independent
fitting sequences run concurrently; `--replica-workers 1` runs them sequentially.
Training order is ICFM, DT, D, MBGAN. Figure order is D, DT, ICFM, MBGAN. Each
fitted model generates 1,000 compositions. Existing trained checkpoints are not
inputs to this workflow. Test rows never enter generator fitting, checkpoint
selection, or a seed search.

| Generator | Prescribed fitting recipe | Training seed | Sampling seed |
| --- | --- | ---: | ---: |
| ICFM | Tree logratios; width-64, three-layer SELU vector field; 20,000 Adam updates; batch 256; rate 0.001 | 20261003 | 2 |
| DT | Independent Beta maximum-likelihood fit at every internal tree node; verified interior solution | deterministic | 1 |
| D | Newton Dirichlet maximum-likelihood fit; score residual below 1e-8 | deterministic | 1 |
| MBGAN | Historical local graph without effective gradient penalty or weight clipping; 500,000 generator updates and five critic updates each; RMSprop rate 0.00005; batch 32 | 1 | 256 |

See [ICFM.md](ICFM.md), [MBGAN.md](MBGAN.md), and
[training_config.json](training_config.json). Dimensions and cohort identity
are read from the external runtime manifest, while fitting settings remain code.
D clips entries to `[1e-10,1]` without subsequent normalization. DT retains
float32 normalization, `1e-7` child-mass smoothing, and positive Beta parameters.

The runtime output contains `replicate_1/`, `replicate_2/`, executable source
snapshots, compact training evidence, and `data/` with copied real inputs plus
four generated matrices. `generation_manifest.json` schema 2 binds runtime
cohort/assets separately from executable source. `COMPLETE` is written only
after both full fitting sequences produce identical parameter arrays and
byte-identical samples. ICFM losses and initialization must agree; MBGAN also
compares every iteration's losses, optimizer state, callbacks, and recorded
Python/NumPy RNG states. Wall-clock timings are excluded. TensorFlow internal
RNG counters are not claimed to support checkpoint resume.

Exact repeatability is scoped to the installed numerical environment. It does
not establish robustness across seeds, adversarial convergence, or generator
fidelity. The inherited local MBGAN variant was originally chosen retrospectively;
this choice remains disclosed and is fixed before each new run.

Pass `/absolute/new-generation/data` as `TWO_SAMPLE_DATA_DIR` to BATTS. Analysis
fits train-versus-generated and test-versus-generated pairs separately. The test
bank is held out from generator fitting; it is used to fit its own density-ratio
comparison. Participant overlap and null diagnostics must be interpreted using
the newly prepared cohort provenance.
