# TwoSample: microbiome case study reproduced from training

Training and analysis workflow for the pooled microbiome case study in
*Two-sample Comparison through Additive Tree Models for Density Ratios*.
The model order throughout is **ICFM, DT, D, MBGAN**.

This release uses the original full pooled data and preprocessing: **1,166
training observations, 292 test observations, and 123 taxa**, from four studies.
It preserves the published sample split and taxon order. Each generator supplies
1,000 compositions. All four generators are fitted using only the 1,166
training observations. ICFM and MBGAN start from seeded random initialization;
DT uses stabilized Beta maximum-likelihood fits, and D uses a Dirichlet
maximum-likelihood fit. Historical neural checkpoints are not inputs to this
workflow. See [generator documentation](generators/README.md).

The analysis uses **BATTS 0.3.0**, pinned to
[`147e511d211919afefb8decac6333e66d24919db`](https://github.com/nawaya040/BATTS/commit/147e511d211919afefb8decac6333e66d24919db).
The MaStatLab fork was checked on 13 October 2026 and remained at `fccebbf`;
its documented upstream released 0.3.0 that day. The BATTS update audit compared
upstream 0.2.2 (`77c2172`) with 0.3.0. Version 0.3.0 changes defaults, so every fit parameter
is explicit: this case study retains **boosting subsample_fraction = 1**.
This run replaces the preceding checkpoint-based neural samples with newly
trained models. Its numerical results must be interpreted from its own tables
and figures. See [verification records](reproducibility/README.md).

## Results and completion

**Status: validated from training through the final report.** Two complete
independent fitting sequences produced identical parameters and synthetic CSVs
for all four generators. All eight comparisons, 24 estimator fits, the null
diagnostic, seven figures, and the ten-page report passed validation.

Read the [final report](results/final/case_study_report.pdf), download the
[regenerated figures](results/final/figures/), or inspect the
[verification records](reproducibility/README.md).

A successful run writes `case_study_report.pdf`, numerical tables, figure data,
software versions, and validation records to `TWO_SAMPLE_RUN_ROOT`. Its report
contains the methods and all seven figures. The final `COMPLETE` marker binds
the validated PDF and report manifest; training alone does not complete the
case study. Verify the released
source, inputs, results, and generator evidence with:

```sh
python3 scripts/verify_from_training_release.py
```

| Figure | Content |
| --- | --- |
| 6 | Test PCoA comparison |
| 7 | Test density-ratio summaries |
| S8 | Training PCoA comparison |
| S9 | Training density-ratio summaries |
| S10 | Training posterior mean and quantile grids |
| S11 | Test posterior mean and quantile grids |
| S12 | Held-out real-data null diagnostic |

## Install

Use R with a C++17 toolchain, and separate Python environments for PyTorch and
TensorFlow. The generator dependency versions are pinned; see the component
README for the tested Python versions, deterministic execution settings, and
training reproduction evidence. ICFM uses CPU execution; MBGAN requires a
working TensorFlow GPU runtime and deterministic GPU operations.

```sh
python3 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan-cuda.txt
python3 -m venv .venv-report
.venv-report/bin/python -m pip install -r requirements-report.txt
Rscript scripts/install_dependencies.R
```

Before running the Python workflow directly, apply the shared-library and
determinism exports in [MBGAN GPU setup](generators/MBGAN.md#gpu-libraries-and-hardware-evidence).
The complete Slurm launcher performs this setup and a two-GPU preflight.

## Reproduce

Run from the repository root. Fit every generator twice, independently, using
the same frozen recipe; generate 1,000 compositions per model in each run;
then verify that fitted parameters and synthetic CSVs agree exactly:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generated
```

Choose a new output directory. The command creates `replicate_1/` and
`replicate_2/`, then writes the six verified matrices and generation evidence
to `output/generated/data/`. It starts without input checkpoints; newly fitted
weights are intermediate outputs. Both independent replicas use CPU ICFM and
GPU MBGAN. With two visible GPUs the replicas run concurrently, one per GPU;
with one GPU they run sequentially. The MBGAN check also
compares every recorded training iteration, both networks, optimizer and RNG
state. Exact agreement is required within the recorded hardware and software
environment; CPU/GPU equality is not assumed. Details are in the
[generator README](generators/README.md).

Compare newly regenerated
matrices, fitted-parameter hashes, and settings directly with the published
reference bank. This checks the released results in addition to agreement
between your two new runs; exact equality is scoped to the recorded environment:

```sh
python3 scripts/verify_training_reference.py output/generated/data
```

Refit all eight BATTS comparisons and the null experiment, render all seven
figures, and build the report:

```sh
export TWO_SAMPLE_DATA_DIR="$PWD/output/generated/data"
export TWO_SAMPLE_RUN_ROOT="$PWD/output/final"
Rscript scripts/reproduce_case_study_batts.R 8
.venv-report/bin/python scripts/build_report.py "$TWO_SAMPLE_RUN_ROOT"
```

The integer selects independent comparison workers; use 1 for serial execution.
Fit caches are bound to the inputs, BATTS commit, settings and fitting source.
The final `COMPLETE` marker requires verified generator provenance, numerical
validation, and PDF validation. Each input directory must include a validated
`generation_manifest.json` and its evidence; old checkpoint-based inputs cannot
be silently substituted. The analysis freezes these records under `generation/`.
For the complete workflow, use [`hpc/from_training.slurm`](hpc/from_training.slurm).
[`hpc/case_study.slurm`](hpc/case_study.slurm) runs the analysis from already
verified generated data. See [cluster execution](hpc/README.md).

The real matrices can also be reconstructed from the four dated
curatedMetagenomicData resources and the exact sample/taxon manifests:

```sh
Rscript scripts/install_preprocessing.R
Rscript scripts/reprocess_microbiome.R
```

This validates every float32 value against the published numerical checksums.
The [preprocessing provenance](data/provenance/README.md) documents the original
filters, unrecovered split seed and ambiguous duplicate source profiles.

## Statistical scope

Each train/test-versus-generator comparison is fitted separately. Test panels
use real test observations in the density-ratio fit; they are not predictions
from a density-ratio estimator fitted to the training comparison. The null
experiment alone evaluates on observations held out from its BATTS fit.

Generator training uses only the 1,166 preserved training observations. Neural
architectures, optimizer settings, training budgets, and seeds are fixed before
test evaluation. The final planned training update supplies each neural model;
test data are not used for fitting, early stopping, or model selection. ICFM
uses 20,000 updates of a three-layer width-64 MLP with Adam (rate 0.001), batch
256, flow noise 0.0001, and training seed 20261003. MBGAN uses 500,000 generator
iterations, five critic updates each, RMSprop (rate 0.00005), batch 32, and
training seed 1, with the gradient penalty connected to critic weights. ICFM
runs on CPU with two numerical threads, while each independent MBGAN run uses
the GPU with two host numerical threads.

Preprocessing was applied collection-wide before the preserved observation
split. The split does not establish participant-level separation, so train-only
generator fitting does not imply independent participants between groups. The
new neural runs are documented reconstructions, not recovery of unknown
historical training trajectories.

Intervals are pointwise posterior intervals from the retained single-chain
protocol. They are not global tests of distributional equality, simultaneous
bands, or a repeated-sampling coverage guarantee. PCoA is descriptive. Matching
generated-row numbers across generators label different generated compositions;
they do not identify the same observation evaluated under all four models.

## Data and attribution

Please cite curatedMetagenomicData and all four source studies listed in
[data/CITATIONS.md](data/CITATIONS.md). See [data terms](data/DATA_TERMS.md) and
[generator notices](generators/README.md). The data are pooled observations
from IBDMDB (931), T2D (277), Hall (209), and Hannigan (41).

The preceding checkpoint-based analysis is preserved separately. This release
contains the validated results from the new training runs.
