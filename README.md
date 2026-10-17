# TwoSample: reproducible case-study code

This repository contains code, configuration, dependency requirements, and
documentation for the corrected microbiome case study: preprocessing, four
generators, synthetic sampling, BATTS fitting, and seven figures. Data,
sample/participant identifiers, tree tables, trained checkpoints, fitted
objects, results, and rendered figures are external runtime files and are not
distributed in the current source tree.

## Prepare inputs

Follow [PREPROCESSING.md](PREPROCESSING.md) to reconstruct the corrected
2,052-observation, 123-taxon cohort with 1,641 training and 411 test observations.
The rule uses full-collection thresholds, keyed study/sample matching, and no
sparsity filter. Exact historical reproduction requires the external dated
source resources, metadata, species universe, taxon/sample ordering, and tree
topology. These data dependencies are listed explicitly in that document;
cloning this code alone does not supply them.

Preparation creates an external directory containing the two real matrices,
their provenance and cohort manifest, and the tree files. Subsequent stages
verify these inputs by hash. Each output command requires a fresh destination.

## Environments

Use separate Python environments for the parametric/ICFM models and historical
TensorFlow MBGAN; their validated NumPy versions differ. Python 3.9.25 was used
for the final experiment. From the repository root:

```sh
python3.9 -m venv .venv-generators
.venv-generators/bin/pip install -r generators/requirements.txt -r requirements-preprocessing.txt
python3.9 -m venv .venv-mbgan
.venv-mbgan/bin/pip install -r generators/requirements-mbgan.txt
Rscript scripts/install_dependencies.R
```

The R installer pins the direct plotting/analysis package versions and requires BATTS 0.3.0 at commit
`147e511d211919afefb8decac6333e66d24919db`; the fitting scripts check that version
and source revision. R 4.6.0 was used for the completed analysis. Use
`Rscript scripts/install_dependencies.R --check` to verify installed direct
dependencies without installing packages. Transitive and system dependencies
must remain compatible. The dated source replay additionally needs
`Rscript scripts/install_preprocessing.R` when downloading through
curatedMetagenomicData. The validated data package was 3.20.0; exact source
hashes and numeric checks are still required. Source-cache reconstruction can
use the existing external RData files without downloading them again.

## Train, sample, fit, and plot

Set absolute paths to the prepared input directory and a new run directory:

```sh
export GENERATOR_PYTHON="$PWD/.venv-generators/bin/python"
export MBGAN_PYTHON="$PWD/.venv-mbgan/bin/python"
export CASE_STUDY_RSCRIPT=Rscript
export WORKERS=8
bash scripts/run_pipeline.sh /absolute/prepared-data /absolute/new-run
```

The runner trains two independent replicas of D, DT, ICFM, and MBGAN, verifies
parameter and sample replay, and generates 1,000 compositions per model. It
then performs eight real-versus-generated comparisons with three estimators
per comparison, runs the held-out null diagnostic, and creates Figures 6, 7,
S8, S9, S10, S11, and S12 under the external analysis directory. Checkpoint
files are produced during training for local replay; they are not published.
Training includes 20,000 ICFM steps and 500,000 MBGAN updates per replica and
can take several hours. See the [generator instructions](generators/README.md)
for separate training/sampling commands and the [Slurm template](hpc/README.md)
for cluster execution.

The generator recipes are in
[training_config.json](generators/training_config.json), the estimator recipes
in [case_study_config.R](scripts/lib/case_study_config.R), and the figure
implementation in [case_study_figures.R](scripts/lib/case_study_figures.R).
ICFM and DT preserve their distinct tree transforms. MBGAN is the historical
local implementation with no effective gradient penalty or weight clipping;
its selection was retrospective in the earlier cohort, before the corrected
cohort was trained. See [MBGAN.md](generators/MBGAN.md) and the bundled third-party
license notices.

To redraw figures from a completed external run, point the plotting scripts at
that run's data and analysis directories:

```sh
export TWO_SAMPLE_DATA_DIR=/absolute/new-run/generators/data
export TWO_SAMPLE_RUN_ROOT=/absolute/new-run/analysis
Rscript scripts/recreate_all_case_study_figures.R
```

The split is by observation and shares participants across training and test
banks. Generator training excludes real test rows; the main test-versus-model
BATTS fits use those test rows themselves. Only the null diagnostic evaluates
observations held out from its BATTS fit. Pointwise intervals and PCoA plots
do not establish global distributional equality.

## Keep the repository code-only

Runtime inputs and outputs belong outside the checkout. Before
committing, check the staged source tree:

```sh
python3 -B scripts/verify_code_only.py
```

The same check runs in GitHub Actions on the committed tree. It rejects data,
tree assets, checkpoints, result tables, figures, binary payloads, symlinks,
and unrecognized release paths. To check an unpacked source archive, use
`--directory`; that mode checks every file on disk. This update removes former
data and figure files from the current branch contents. Earlier public Git
commits remain in history.
