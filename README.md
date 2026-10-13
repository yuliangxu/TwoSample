# TwoSample: microbiome case study reproduced from training

Training and analysis workflow for the pooled microbiome case study in
*Two-sample Comparison through Additive Tree Models for Density Ratios*.
Figures, legends, and the report use **D, DT, ICFM, MBGAN**. The recorded fitting
sequence remains ICFM, DT, D, MBGAN.

This workflow preserves the original full pooled data and preprocessing:
**1,166 training observations, 292 test observations, and 123 taxa**, from four
studies. Each generator supplies 1,000 compositions. All four generators use
only the training observations. ICFM and MBGAN start from seeded random
initialization; DT uses stabilized Beta maximum-likelihood fits and D uses a
Dirichlet maximum-likelihood fit. Historical neural checkpoints are not inputs.
See [generator documentation](generators/README.md).

## Selected final code

The **historical local MBGAN implementation** is the selected final code, with
its diagnostic callback randomness made reproducible. Implementation checks
passed: two independent 120-iteration fits agreed exactly, and the wrapper
matched the original training loop's parameters and optimizer states.
See the [implementation validation](reproducibility/historical_mbgan_implementation_validation.json).

**Full 500,000-iteration reproduction and updated results are pending.** Job
58514810 runs two complete fitting sequences, followed by BATTS and report
generation. This code release contains the preserved real inputs and the
selected workflow; it does not contain synthetic matrices or reports from a
different MBGAN implementation. The preceding validated results remain available
at [commit b009075](https://github.com/yuliangxu/TwoSample/tree/b00907514c3bbb7ab17c4eeacc5091ab4f441d8a).
The [code release manifest](code_release_manifest.json) records this release's
scope and file hashes. Verify the packaged code and inputs with:

```sh
python3 scripts/verify_code_release.py
```

The earlier `ibdmdb_case_study_report.pdf` used a fresh 500,000-iteration fit
with 945 training observations, 236 test observations, and 77 taxa. The earlier
`updated_generators_case_study_report.pdf` used the full pooled 123-taxon data
and replayed an older MBGAN checkpoint. Its MBGAN was not newly trained. The
restoration fits the historical local variant on the full pooled training set;
its results need not match either earlier report.

The restored variant has **no effective gradient penalty and no weight
clipping**. It preserves the historical Keras 2.13.1 graphs and CPU training
recipe; it should not be described as the published WGAN-GP implementation.
Its former diagnostic callback reseeded NumPy from system entropy. The new
callback uses an isolated seeded random stream, defining a reproducible new
trajectory rather than recovering the historical weights. Reproducibility and
favorable plots alone would not establish training stability or distributional
agreement. See [MBGAN methods and provenance](generators/MBGAN.md).

The analysis uses **BATTS 0.3.0**, pinned to
[`147e511d211919afefb8decac6333e66d24919db`](https://github.com/nawaya040/BATTS/commit/147e511d211919afefb8decac6333e66d24919db).
The MaStatLab fork was checked on 13 October 2026 and remained at `fccebbf`;
its documented upstream released 0.3.0 that day. The update audit compared
upstream 0.2.2 (`77c2172`) with 0.3.0. All fitting parameters are explicit,
including the retained **boosting subsample_fraction = 1**. This restoration
preserves the other generators, real-data preprocessing, split, and BATTS recipe.

## Results and completion

A completed run contains eight comparisons, 24 estimator fits, the null
diagnostic, seven figures, and a ten-page report. Its final `COMPLETE` marker
requires numerical, provenance, and PDF validation, in addition to training.
The new report and figures will be exported only after those checks pass.
Current [verification records](reproducibility/README.md) distinguish the passed
implementation checks from the pending complete run.

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

Use R with a C++17 toolchain and separate Python environments for PyTorch,
TensorFlow, and report rendering. The tested generator Python version is
3.9.25. ICFM and restored MBGAN use CPU execution with two numerical threads.
The MBGAN environment pins TensorFlow CPU 2.13.1 and Keras 2.13.1.

```sh
python3 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan.txt
python3 -m venv .venv-report
.venv-report/bin/python -m pip install -r requirements-report.txt
Rscript scripts/install_dependencies.R
```

## Reproduce

Run from the repository root. Fit every generator twice independently, generate
1,000 compositions per model in each run, then compare fitted parameters and
synthetic CSVs exactly:

```sh
.venv-generators/bin/python -m generators.reproduce \
  --tensorflow-python .venv-mbgan/bin/python \
  --output output/generated
```

Choose a new output directory. Two CPU replicas run concurrently by default;
add `--replica-workers 1` to run them sequentially. The command creates
`replicate_1/` and `replicate_2/`, then writes the six verified matrices and
training evidence to `output/generated/data/`. Newly fitted weights are
intermediate outputs, not input checkpoints. MBGAN validation compares every
recorded training iteration, generator and critic arrays, optimizer arrays,
and the explicitly recorded Python and NumPy random states. TensorFlow graph
seeds and update counts are recorded, without claiming to checkpoint its
internal random-operation counters. Exact equality is scoped to the recorded
hardware and software environment.

After the completed results have been exported, compare a newly generated bank
with its reference data and parameter hashes:

```sh
python3 scripts/verify_training_reference.py output/generated/data
```

This reference comparison is unavailable in the present code-only release;
the generation command itself still checks exact agreement between its two
fresh replicas.

Refit all eight BATTS comparisons and the null experiment, render all seven
figures, and build the report:

```sh
export TWO_SAMPLE_DATA_DIR="$PWD/output/generated/data"
export TWO_SAMPLE_RUN_ROOT="$PWD/output/final"
Rscript scripts/reproduce_case_study_batts.R 8
.venv-report/bin/python scripts/build_report.py "$TWO_SAMPLE_RUN_ROOT"
```

The integer selects comparison workers; use 1 for serial execution. Fit caches
are bound to the inputs, BATTS commit, settings, and fitting source. Every
input directory must include its validated `generation_manifest.json` and
evidence. The analysis freezes them under `generation/`. The complete cluster
workflow is [`hpc/from_training.slurm`](hpc/from_training.slurm);
[`hpc/case_study.slurm`](hpc/case_study.slurm) analyzes already verified generated
data. See [cluster execution](hpc/README.md).

The real matrices can also be reconstructed from the four dated
curatedMetagenomicData resources and the exact sample/taxon manifests:

```sh
Rscript scripts/install_preprocessing.R
Rscript scripts/reprocess_microbiome.R
```

This validates every float32 value against the preserved numerical checksums.
The [preprocessing provenance](data/provenance/README.md) documents the filters,
unrecovered split seed, and ambiguous duplicate source profiles. Once the new
release is exported, verify its source, inputs, results, and evidence with:

```sh
python3 scripts/verify_from_training_release.py
```

## Statistical scope

Each training/test-versus-generator comparison is fitted separately. Test panels
use real test observations in the density-ratio fit; they are not predictions
from the density-ratio fit to the training comparison. The null experiment alone
evaluates observations held out from its BATTS fit.

Generators use only the 1,166 preserved training observations. The restored
MBGAN recipe fixes seed 1, 500,000 generator iterations, five critic updates per
iteration, RMSprop at rate 0.00005, batch 32, and the final update. No new
seed or checkpoint search uses the test results. However, returning to this
historical implementation after inspecting previous reports is a retrospective
method choice; it is not a blind, prespecified comparison of MBGAN variants.
ICFM retains 20,000 updates of a three-layer width-64 MLP with Adam at rate
0.001, batch 256, flow noise 0.0001, and training seed 20261003.

Preprocessing was applied collection-wide before the preserved observation
split. The split does not establish participant-level separation. The new
neural fits do not recover unknown historical training trajectories.

Intervals are pointwise posterior intervals from the retained single-chain
protocol, not global distributional-equality tests, simultaneous bands, or a
repeated-sampling coverage guarantee. PCoA is descriptive. Matching generated-row
numbers across models identify different compositions.

## Data and attribution

Please cite curatedMetagenomicData and the four source studies listed in
[data/CITATIONS.md](data/CITATIONS.md). See [data terms](data/DATA_TERMS.md) and
[generator notices](generators/README.md). The pooled data contain observations
from IBDMDB (931), T2D (277), Hall (209), and Hannigan (41).

Earlier checkpoint-based and connected-gradient-penalty results are preserved
separately from this selected implementation.
