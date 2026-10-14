# Verification records

The **historical local MBGAN variant is the selected final implementation**.
The [implementation validation](historical_mbgan_implementation_validation.json)
records passed 120-iteration checks: two independent fits agreed exactly, and
the wrapper matched the original training loop's parameters and optimizer
states. The [provenance audit](historical_mbgan_provenance.json) identifies the
historical code and the two earlier reports.

**Full reproduction and updated results passed** in job 58514810, which
completed with exit code 0 after 9 hours, 4 minutes, 29 seconds. Both independent
fitting sequences reproduced all four models exactly, including both complete
500,000-iteration MBGAN trajectories. The BATTS analysis passed all eight
comparisons and 24 estimator fits; all 292 null intervals cover zero. Seven
figures and a ten-page report passed numerical, provenance, and PDF checks.
The final report was generated on 14 October 2026 at 03:56 UTC
(13 October at 23:56 EDT).

The [full validation](historical_mbgan_full_validation.json) records these
checks. The [release manifest](../release_manifest.json) binds the packaged
source, six input matrices, evidence, and completed results. Verify the release
with:

```sh
python3 scripts/verify_from_training_release.py
```

The [report](../results/final/case_study_report.pdf) and
[figures](../results/final/figures/) use the newly trained MBGAN bank bound by
[data/generation_manifest.json](../data/generation_manifest.json) and
[data/evidence/](../data/evidence/).

The restored workflow repeats all four fits from scratch with CPU ICFM and
historical local CPU MBGAN. MBGAN uses TensorFlow CPU 2.13.1 and Keras 2.13.1,
with isolated seeded diagnostic randomness. Both full replicas reproduced
exactly within their recorded hardware and software environment. The real data,
preprocessing, split, other generators, and BATTS recipe remain unchanged.

The earlier IBDMDB report did fully retrain the historical local MBGAN on 945
training rows with 77 taxa; its callback nevertheless reseeded NumPy from system
entropy. The earlier pooled 123-taxon report replayed an old MBGAN checkpoint.
The new pooled fit must be evaluated on its own evidence. Selecting this
historical implementation after viewing previous reports is a retrospective
choice, even though the new fit keeps its fixed seed and final iteration.

## Required evidence

| Record | What it verifies |
| --- | --- |
| Generation output `execution.json` | Frozen settings, source hashes, interpreters, two independent fits, and no pretrained inputs |
| Generation output `data/generation_manifest.json` | Training/test hashes, four completed train-only fits, source and metadata hashes, and sample hashes |
| Generation output `data/evidence/*_training.json` | Model-specific configuration, completion, input identity, and native training metadata |
| Generation output `data/evidence/reproduction.json` | Exact fitted-parameter and synthetic-sample equality across both complete runs |
| Generation output `data/evidence/mbgan_independent_validation.json` | Every recorded iteration's losses, generator/critic and optimizer arrays, explicit Python/NumPy states, TensorFlow seed/update-count provenance, and samples |
| Generation output `data/evidence/runtime_environment.json` | Actual CPU hardware, interpreters, dependencies, numerical execution settings, and preflight checks |
| Analysis output `input_manifest.json` and `generation/` | Frozen six matrices, analysis source, and validated generator evidence |
| Analysis output `analysis_validation.json` | Eight comparisons, 24 estimator fits, null intervals, figures, and generator provenance |
| Analysis output `report_manifest.json` and `COMPLETE` | Ten-page PDF, seven embedded figures, numerical checks, rendering, and matching provenance |

The [preprocessing validation](preprocessing_validation.json) independently
reconstructs the preserved real matrices. The
[BATTS update audit](BATTS_update_audit.json) identifies upstream 0.3.0 and its
pinned commit. Those records remain applicable. The completed restored run has
its own numerical, provenance, and report checks in the full validation record.
The required presentation order remains **D, DT, ICFM, MBGAN**.

## Reproduce and verify

Run the full independent generation workflow:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generated
```

The default is two concurrent CPU replicas; add `--replica-workers 1` for
sequential fitting. To revalidate the completed bank and evidence:

```sh
.venv-generators/bin/python -m generators.validate \
  --data-dir output/generated/data \
  --output output/generated_validation
```

Generation and validation destinations must be new. Compare newly generated
data with the released reference CSVs,
configurations, fitted-parameter hashes, and recorded states:

```sh
python3 scripts/verify_training_reference.py output/generated/data
```

The completed reference bank is included in `data/`. The full generation
workflow independently compares its two fresh replicas without requiring a
released reference bank.

The generation gate requires exact equality. MBGAN trajectory comparison excludes
wall-clock timings. Arrays and explicit states are compared directly or through
deterministic hashes; checkpoint-container bytes may contain run-specific
metadata. TensorFlow graph seeds and update counts are recorded, but internal
random-operation counters are not claimed to be checkpointed. The workflow
requires finite losses and final arrays, without claiming that every gradient
was separately inspected. Repeatability does not establish stability across
seeds or generator fidelity.

Dependencies are pinned in the component requirement files. BATTS is pinned by
source commit; R versions and session information are recorded. Exact agreement
is required within the recorded environment, without assuming the same bytes
across processors, compilers, thread counts, or library versions.

## Release validation

The exporter `scripts/export_training_release.py RUNROOT` accepts completed
`generators/` and `analysis/` subdirectories. It requires completion markers and
matching training/analysis provenance before curating a replacement release.
The resulting `release_manifest.json` binds source, inputs, evidence, and results.
Verify the exported release with:

```sh
python3 scripts/verify_from_training_release.py
```

`verify_release.py` is retained as part of the older executed source snapshot;
use `verify_from_training_release.py` for this completed from-training release.
The selected code before full-run completion remains at
[commit fc7f6fa](https://github.com/yuliangxu/TwoSample/tree/fc7f6fa21e3b834b3be1ff459a473be9c7f30c88).
The preceding connected-gradient-penalty run's data and results remain at
[commit b009075](https://github.com/yuliangxu/TwoSample/tree/b00907514c3bbb7ab17c4eeacc5091ab4f441d8a).
