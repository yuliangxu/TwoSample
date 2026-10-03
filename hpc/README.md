# Full-study IBDMDB experiment

This workflow applies `scripts/microbiome_subset.R`'s filters to **only the full
HMP_2019_ibdmdb study**, then creates an 80/20 observation split, trains all four
supplied generators on the training compositions, and produces Figures 6, 7,
S8, S9, S10, S11 and S12 with BATTS. It does not use the previously audited
931-observation subset or the bundled generator checkpoints.

The pinned resource is `2021-10-14.HMP_2019_ibdmdb.relative_abundance` from
curatedMetagenomicData 3.20.0. A local preprocessing check retained **1,181
observations and 77 taxa**, giving **945 training and 236 test observations**
with split seed 20261003. Dimensions are calculated from the resource, and
the phylogeny is pruned to the selected taxa; no 123-taxon assumption remains.

The original rules are applied in their original order within this study:

1. Identify samples with total relative abundance >= 99.
2. Retain taxa whose mean abundance among these complete samples is >= 0.1.
3. Retain complete samples with **more than** 10% zero retained taxa and a
   retained-taxon mean >= the mean across all original study observations.
4. Normalize each retained composition over retained taxa, round to float32,
   and split observations 80/20 using a seeded permutation.

Preprocessing precedes splitting, as in the supplied workflow. Repeated
observations from one participant may appear in both splits. Generator fitting
uses only the training CSV; test observations are used for evaluation.

## Install on the cluster

Run from the repository root on a node with internet access. Load your site's
R 4.6.x, Python 3.9 and C/C++ compiler modules first. R dependencies use
Bioconductor 3.23; BATTS is pinned to source commit
`77c217297910a5ba50b71313e8289d9024b669c9` (version 0.2.2).

```bash
Rscript scripts/setup_ibdmdb.R --lib-dir=output/ibdmdb_study/r_lib
python3.9 -m venv .venv-generators
.venv-generators/bin/python -m pip install -r generators/requirements.txt
python3.9 -m venv .venv-mbgan
.venv-mbgan/bin/python -m pip install -r generators/requirements-mbgan.txt
```

The TensorFlow environment is separate because its NumPy requirements differ
from the other generators. Environment files, data, checkpoints and figures
are generated locally and excluded from Git under `output/` and `.venv-*`.

If compute nodes have no internet access, populate an ExperimentHub cache and
prepare the study on a login/data-transfer node first:

```bash
export EXPERIMENT_HUB_CACHE="$PWD/output/experimenthub_cache"
python3 scripts/run_ibdmdb_experiment.py --stage prepare --cache-dir "$EXPERIMENT_HUB_CACHE"
export IBDMDB_OFFLINE=1
```

Alternatively, omit the preparatory command and allow the preparation job to
download the resource. The default R library is `output/ibdmdb_study/r_lib`;
`--r-lib` / `IBDMDB_R_LIB` can point to a different installation.

## Run with Slurm

Edit `hpc/ibdmdb_job.sh` for your site's account, partition, module loads,
memory and time limits. Its CPU template requests 8 CPUs, 32 GB and seven days
per job; the MB-GAN job can require substantial time at the supplied budget.
Submit from the repository root:

```bash
bash hpc/submit_ibdmdb.sh
```

The dependency chain is preparation, a four-job generator array, then BATTS,
all seven figures and the report. Generator defaults are **20,000 ICFM steps**,
**500,000 MB-GAN iterations**, and **1,000 synthetic observations per model**.
The original generator training defaults and sampling seeds are retained.
Every generator receives this experiment's training CSV explicitly; both
neural samplers receive their newly trained checkpoints explicitly.

Set `IBDMDB_OUTPUT` to an absolute scratch-directory path to relocate the
experiment. Optional environment overrides are `GENERATOR_PYTHON`,
`MBGAN_PYTHON`, `IBDMDB_RSCRIPT`, `DRIVER_PYTHON`, `GENERATOR_THREADS`,
`ICFM_STEPS`, `MBGAN_ITERATIONS` and `N_SYNTHETIC`. These must be consistent
across the jobs. Defaults use CPUs. For GPUs, configure the Slurm resources
and a compatible cluster GPU environment, then set `ICFM_DEVICE=cuda:0`
and/or `MBGAN_DEVICE=gpu` for the corresponding job.

For another scheduler, invoke the same driver stages manually:

```bash
python3 scripts/run_ibdmdb_experiment.py --stage prepare
python3 scripts/run_ibdmdb_experiment.py --stage d
python3 scripts/run_ibdmdb_experiment.py --stage dt
python3 scripts/run_ibdmdb_experiment.py --stage icfm
python3 scripts/run_ibdmdb_experiment.py --stage mbgan
python3 scripts/run_ibdmdb_experiment.py --stage batts --workers 8
python3 scripts/run_ibdmdb_experiment.py --stage figures
python3 scripts/run_ibdmdb_experiment.py --stage report
```

`--stage all` runs the entire workflow sequentially. Use `--help` for path,
seed, device and budget overrides. For a smoke check, use a **new output
directory** and consistently pass `--icfm-steps 4 --mbgan-iterations 2
--n-synthetic 32` to the preparation and generator stages. These shortened
budgets verify execution only; they are not the experiment settings.

Completed preparation/generator stages are reused only when their recorded
settings and output hashes agree. BATTS estimator fits are independently
cached by input hashes and fit settings. Neural training does not resume
mid-training: if an incomplete generator directory remains after a failed
job, preserve it and select a new output directory for a fresh run.

## Outputs and validation

The experiment directory contains the full filtering audits, train/test
sample IDs, ordered taxa, the pruned binary tree, four fresh generator
checkpoints, eight real-versus-generated comparison tables, the held-out
real-versus-real null experiment, seven PDFs, summary CSVs and `REPORT.md`.
`experiment_provenance.json` records inputs, settings and figure hashes;
`COMPLETE` appears only after the report and all seven PDFs are validated.

BATTS comparisons retain 200 trees, 2,000 burn-in iterations, 1,000 retained
iterations and the original boosting/CV settings. Train and test comparisons
are fitted separately, following the existing code. The null fit uses 500
burn-in and 500 retained iterations and evaluates all real test observations.
The driver checks study membership, disjoint sample IDs, composition sums,
dimensions and checkpoint hashes, and refuses changed data in a completed run.

The local code check covers full-study preprocessing and fresh fitting/sampling
for all four generators with shortened neural budgets. Full neural training
and the full study-only BATTS experiment are intended to run on the cluster.

The integrity and Slurm command tests run without downloaded data or R:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -p 'test_ibdmdb*.py' -v
```
