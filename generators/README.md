# Reproduce every generator from training data

This workflow fits **ICFM, DT, D, MBGAN**, in that order, using only the preserved
1,166 training observations with 123 taxa. Figures and reports display
**D, DT, ICFM, MBGAN**. Each model supplies 1,000 synthetic compositions. Neural
models start from seeded random initialization; no historical checkpoint is an
input. The 292 real test observations are copied for subsequent BATTS analysis
and do not enter generator fitting or epoch selection.

**Selected final MBGAN implementation: historical local variant.**
[Implementation checks](../reproducibility/historical_mbgan_implementation_validation.json)
passed for two independent 120-iteration fits and parity with the original
training loop. **Full reproduction and updated analysis also passed** in job
58514810: both fitting sequences reproduced all four models exactly, including
both 500,000-iteration MBGAN trajectories. This release includes the six
reference matrices, [generation manifest](../data/generation_manifest.json),
[training evidence](../data/evidence/), [final report](../results/final/case_study_report.pdf),
and [regenerated figures](../results/final/figures/). See the
[full validation](../reproducibility/historical_mbgan_full_validation.json).

| Model | Frozen fitting recipe | Training seed | Sampling seed |
| --- | --- | ---: | ---: |
| ICFM | Three hidden layers of width 64; 20,000 Adam updates at rate 0.001; batch 256; flow noise 0.0001; final update | 20261003 | 2 |
| DT | Independent positive-parameter Beta maximum-likelihood fits at all 122 tree nodes | deterministic fit | 1 |
| D | Newton Dirichlet maximum-likelihood fit | deterministic fit | 1 |
| MBGAN | Historical local graph without effective gradient penalty or clipping; 500,000 generator iterations; five critic updates each; RMSprop at rate 0.00005; batch 32; final iteration | 1 | 256 |

[training_config.json](training_config.json) records the settings. ICFM and MBGAN
use CPU execution with two numerical threads. MBGAN preserves the historical
Keras 2.13.1 compiled graphs and removes the diagnostic callback's entropy
reseeding by isolating its random draws. This defines a new reproducible
trajectory and does not promise the old weights or favorable results. See
[ICFM.md](ICFM.md) and [MBGAN.md](MBGAN.md).

## Install and run

Use separate environments for the pinned PyTorch and TensorFlow packages. The
reference Python version is 3.9.25; MBGAN uses TensorFlow CPU 2.13.1, Keras
2.13.1, and NumPy 1.24.3.

```sh
python3 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan.txt
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generated
```

Run from the repository root and choose a new output directory. Two complete
CPU fitting and sampling sequences run concurrently by default; add
`--replica-workers 1` for sequential execution. Both start afresh with the same
settings and seeds. Newly fitted weights are intermediate outputs for sampling
and auditing.

The output layout is:

- `replicate_1/` and `replicate_2/`: fitted parameters, samples, native training
  metadata, and logs for all four generators.
- `execution.json`: frozen settings, source hashes, interpreter paths, and the
  empty list of pretrained-model inputs.
- `data/`: unchanged real training/test CSVs, four new synthetic CSVs,
  `generation_manifest.json`, and compact verification records in `evidence/`.
- `source/`: generator source and fitting-asset snapshot.
- `mbgan_independent_validation.json`: complete MBGAN trajectory comparison.
- `COMPLETE`: written only after both runs satisfy all generation checks.

Use `output/generated/data` as `TWO_SAMPLE_DATA_DIR` for BATTS. Generator
completion attests to training and sampling; the analysis has a separate
completion marker requiring numerical and PDF validation.

Compare a newly generated bank with the released reference, including all six
CSVs, fitted parameter hashes,
configurations, and recorded neural states:

```sh
python3 scripts/verify_training_reference.py output/generated/data
```

The reference bank is included in `data/`. The generation workflow also
compares its two independent replicas without requiring a released reference
bank. Verify the packaged source, data, evidence, and results with:

```sh
python3 scripts/verify_from_training_release.py
```

## Required reproduction checks

Both complete runs must produce identical fitted parameter arrays and
byte-identical synthetic CSVs for every model. The verifier also checks matrix
dimensions, finite nonnegative entries, row sums, training input hashes, source
hashes, and completion of the full training budgets. No numerical tolerance
replaces exact equality in this gate.

For ICFM, initial parameter hashes and every recorded loss must agree; final
parameters must agree after changing through training. For MBGAN, the verifier
compares every iteration's recorded losses, all generator and critic tensors,
optimizer arrays, and the explicitly saved Python and NumPy random states.
TensorFlow graph seeds and update counts are recorded as provenance; internal
TensorFlow random-operation counters are not claimed to be checkpointed.
Wall-clock timings are excluded. Training requires finite losses and final
arrays, changed parameters, and exact save/reload sampling. These checks do not
assert that every individual gradient was inspected.

To revalidate a completed bank:

```sh
.venv-generators/bin/python -m generators.validate \
  --data-dir output/generated/data \
  --output output/generated_validation
```

The validation destination must not already contain `validation.json`.
`manifest.json` binds the preserved real data and fitting assets. The generated
`data/generation_manifest.json` binds training records, reproduction evidence,
and sample hashes. Exact reproduction is scoped to the recorded environment;
equality across TensorFlow versions, CPU architectures, or thread settings is
not assumed. Two matching fixed-seed fits establish repeatability under these
conditions, not stability across seeds or proof of generator fidelity.

## Provenance, definitions, and attribution

The earlier IBDMDB report fitted the historical local MBGAN from scratch for
500,000 iterations on 945 training observations with 77 taxa (236 test rows).
The earlier full pooled report used 1,166 training rows and 123 taxa but replayed
an old MBGAN checkpoint. This restoration trains the historical variant on the
full pooled data; it need not reproduce either report. Its diagnostic callback
now avoids the old `np.random.seed(None)` entropy reset.

The final training update and original seeds are retained without a new
test-based seed or checkpoint search. Choosing this implementation after seeing
earlier reports is nevertheless retrospective and must not be described as a
blind, prespecified comparison of MBGAN variants.

D clips entries to `[1e-10, 1]` without renormalization, preserving the recovered
Newton fit; its likelihood-score residual must be below `1e-8`. DT preserves
float32 row normalization and `1e-7` child-mass smoothing, fits positive Beta
parameters at all 122 internal nodes, and requires an interior solution with
residual below `1e-8` at every node. Both retain the original NumPy `RandomState`
sampling order.

The original preprocessing, real-data split, and taxon order are preserved.
Preprocessing was collection-wide; the observation split does not establish
participant-level separation. The torchdyn subset retains its Apache-2.0
license and source manifest in `vendor/`. MBGAN retains the GPL-3.0 and credited
MIT notices in `licenses/`; these do not relicense unrelated repository files.
