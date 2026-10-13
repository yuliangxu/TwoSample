# Cluster execution

Install the pinned environments and BATTS commit as described in the root
README. Adjust the account, partition, and resources for your cluster before
submitting from the repository root.

## Complete training and analysis

[`from_training.slurm`](from_training.slurm) fits all four generators twice
from their real training data. The two complete sequences run concurrently,
using two numerical threads each. Both ICFM and the restored historical MBGAN
implementation run on CPU. No input checkpoints are used. MBGAN uses the
pinned TensorFlow CPU 2.13.1 and Keras 2.13.1 environment, with 500,000 generator
iterations per replica. The reference allocation is eight CPUs, 32 GB RAM,
and twenty-four hours on the `common` partition; it requests no GPUs.

The site-specific launcher freezes a new source snapshot, writes all source
hashes and the exact launch command, and submits that snapshot:

```sh
python3 hpc/launch_cluster.py \
  --run-root /cwork/yx306/TwoSample-historical-mbgan-full-20261013
```

The run directory must be new and outside the checkout. To stage the run for
review without submission, add `--stage-only`; `submission.json` contains the
complete argument list for `sbatch`. The launcher uses the existing historical
TensorFlow environment and the recorded R and report environments on this
cluster. Existing runs and their checkpoints remain untouched.

For another cluster, install the pinned environments and submit the template:

```sh
export GENERATOR_PYTHON="$PWD/.venv-generators/bin/python"
export MBGAN_PYTHON="$PWD/.venv-mbgan/bin/python"
export REPORT_PYTHON="$PWD/.venv-report/bin/python"
export CASE_STUDY_RSCRIPT=Rscript
export TWO_SAMPLE_EXECUTION_ROOT=/path/to/new/durable/execution
export TWO_SAMPLE_GENERATION_ROOT=/path/to/new/durable/generation
export TWO_SAMPLE_RUN_ROOT=/path/to/new/durable/analysis
sbatch hpc/from_training.slurm
```

[`runtime_preflight.py`](runtime_preflight.py) verifies the pinned CPU
packages, training configuration, thread allocation, and staged source hashes.
It records TensorFlow build details, the CPU model and flags, affinity, all
installed Python packages, environment settings, and a CPU calculation check
in `runtime_environment.json`. The generation manifest binds this evidence by
hash. The recorded configuration includes the historical model variant and
callback RNG policy, with hashes of the full dependency pins and training
configuration. oneDNN and TensorFlow deterministic operations are enabled,
matching the trainer's runtime settings. Exact reproduction is checked within
the recorded environment; changing
CPU hardware or numerical libraries can change training results.

Analysis starts only after both complete training sequences produce identical
parameters and samples and write the generation `COMPLETE` marker. The
analysis input gate checks the generation manifest and its evidence again.
Then BATTS runs eight comparisons and 24 estimator fits, the null diagnostic,
and seven figures, followed by the report. Report rendering requires
`ANALYSIS_COMPLETE`; the job succeeds only after the final report `COMPLETE`
marker exists. Check the scheduler's final exit status as well.

## Analysis from verified generated data

[`case_study.slurm`](case_study.slurm) runs the BATTS analysis and report using
an already completed generation run:

```sh
export TWO_SAMPLE_DATA_DIR=/path/to/completed/generation/data
export TWO_SAMPLE_RUN_ROOT=/path/to/new/durable/analysis
export REPORT_PYTHON="$PWD/.venv-report/bin/python"
sbatch hpc/case_study.slurm
```

This template uses `Rscript` from `PATH`, requests eight CPUs and 64 GB RAM,
and limits numerical libraries to one thread per analysis worker. All six
CSVs, the from-training generation manifest, and its evidence must be present.
Earlier checkpoint-replay inputs fail the provenance gate.
