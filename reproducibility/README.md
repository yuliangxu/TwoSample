# Verification records

The full from-training run passed validation: all four generator pairs matched
exactly after complete independent fitting, and all eight comparisons,
24 estimator fits, the null diagnostic, seven figures and ten-page report
passed their checks. See [the release validation summary](from_training_validation.json)
and [the generation manifest](../data/generation_manifest.json).

The new workflow repeats the complete fitting sequence twice with CPU ICFM
and GPU MBGAN in each replica. The earlier CPU MBGAN run is retained separately.
The GPU pair reproduced exactly within its recorded hardware and software
environment; equality between CPU and GPU trajectories is not assumed.

The [independent report audit](report_independent_validation.json) recomputes the
summary tables, checks all figure data and model ordering, and records visual
review of all seven figures and ten report pages. It also lists the numerical
changes that must be reflected in the manuscript.

The completed workflow retains the following evidence:

| Record | What it verifies |
| --- | --- |
| Generation output `execution.json` | Frozen training configuration, source hashes, interpreters, two independent runs, and no pretrained inputs |
| Generation output `data/generation_manifest.json` | Training/test input hashes, four completed train-only fits, source hashes, metadata hashes, and synthetic output hashes |
| Generation output `data/evidence/*_training.json` | Model-specific configuration, completion, training input, and native training metadata |
| Generation output `data/evidence/reproduction.json` | Exact fitted-parameter and synthetic-sample equality across both complete runs |
| Generation output `data/evidence/mbgan_independent_validation.json` | MBGAN losses and gradient diagnostics for every iteration, generator/critic tensors, optimizer state, RNG state, and samples |
| Generation output `data/evidence/runtime_environment.json` | Actual GPU hardware, driver, TensorFlow build, pinned and loaded CUDA libraries, and per-device preflight checks |
| Analysis output `input_manifest.json` and `generation/` | Frozen six matrices, analysis source, and validated generator evidence |
| Analysis output `analysis_validation.json` | Eight comparisons, 24 estimator fits, null intervals, figures, and generator provenance |
| Analysis output `report_manifest.json` and `COMPLETE` | Ten-page PDF, seven embedded figures, numerical checks, rendering, and matching provenance |

The existing preprocessing validation documents independent reconstruction of
the preserved real matrices. The BATTS audit identifies upstream 0.3.0 and its
pinned source commit. Both remain relevant to the new workflow, whose neural
samples and resulting BATTS tables are evaluated separately.

To run the complete generator verification from code:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generated
```

To revalidate its completed data and evidence:

```sh
.venv-generators/bin/python -m generators.validate \
  --data-dir output/generated/data \
  --output output/generated_validation
```

Compare the regenerated bank directly with the published reference CSVs,
configurations, fitted-parameter hashes, and neural state hashes:

```sh
python3 scripts/verify_training_reference.py output/generated/data
```

The generation and validation destinations must be new. The full reproduction
check requires exact equality, rather than agreement within numerical
tolerances. The MBGAN trajectory check excludes elapsed wall-clock time.
Parameter arrays and states are compared directly or through deterministic
array hashes; checkpoint container bytes may include run-specific metadata.

Python numerical and reporting dependencies are pinned in the component
requirements files. BATTS is pinned by source commit; R versions and session
information are recorded by the analysis. Exact agreement is required between
the two independent runs in the recorded environment, without assuming the
same bytes across different hardware, compilers, or numerical-library versions.

The public `release_manifest.json` binds the curated source, inputs, evidence, and result
files. Verify that package using the standard-library command:

```sh
python3 scripts/verify_from_training_release.py
```

The exporter `scripts/export_training_release.py RUNROOT` accepts a completed
run with `generators/` and `analysis/` subdirectories. It requires completion
markers and matching training/analysis provenance before curating the release.
The original `verify_release.py` remains unchanged as part of the executed
source snapshot; use `verify_from_training_release.py` for this release.
