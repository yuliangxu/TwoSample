# Cluster execution

Install the pinned environments and BATTS commit as described in the root
README. Edit account, partition, module loads, and resource limits for your
cluster before submitting from the repository root.

## Complete training and analysis

[`from_training.slurm`](from_training.slurm) fits all four generators twice,
verifies exact reproduction, then runs all BATTS comparisons, the null
experiment, seven figures, and the report. Each independent sequence uses CPU
ICFM and GPU MBGAN; the job requires a TensorFlow-compatible GPU allocation.
It requires no input checkpoints. The reference launcher requests two A5000
GPUs, discovers the pinned NVIDIA library directories, sets deterministic
runtime variables, and requires both allocated GPUs to pass preflight.
For example:

```sh
export GENERATOR_PYTHON="$PWD/.venv-generators/bin/python"
export MBGAN_PYTHON="$PWD/.venv-mbgan/bin/python"
export REPORT_PYTHON="$PWD/.venv-report/bin/python"
export CASE_STUDY_RSCRIPT=Rscript
export TWO_SAMPLE_GENERATION_ROOT=/path/to/new/durable/generation
export TWO_SAMPLE_RUN_ROOT=/path/to/new/durable/analysis
sbatch hpc/from_training.slurm
```

Use a new generation directory: both full independent runs start from scratch.
This two-GPU job runs the independent replicas concurrently, one per GPU,
using two numerical threads for CPU
ICFM and two host numerical threads for each GPU MBGAN run. Configure the GPU
request and device visibility for your cluster, and use the same recorded GPU
hardware and runtime for both MBGAN replicas. The workflow then sets
`TWO_SAMPLE_DATA_DIR` to `TWO_SAMPLE_GENERATION_ROOT/data`. The BATTS analysis
uses CPU workers and needs no GPU. Adjust CPU, GPU, memory, and time limits to
the launcher and cluster; check the scheduler's final exit status. Exact
reproduction is checked within the recorded environment, without assuming
matching results between CPU and GPU execution.

For a one-GPU allocation, use the direct Python command documented in the
[generator README](../generators/README.md), after the [GPU library
setup](../generators/MBGAN.md#gpu-libraries-and-hardware-evidence). It runs two
fresh replicas sequentially. The supplied Slurm launcher itself requires two
GPUs and records their runtime details.

## Analysis from verified generated data

[`case_study.slurm`](case_study.slurm) runs only the BATTS analysis and report.
First complete generator reproduction, then point the job at its verified data:

```sh
export TWO_SAMPLE_DATA_DIR=/path/to/completed/generation/data
export TWO_SAMPLE_RUN_ROOT=/path/to/new/durable/analysis
export REPORT_PYTHON="$PWD/.venv-report/bin/python"
sbatch hpc/case_study.slurm
```

This template uses `Rscript` from `PATH`, requests eight CPUs and 64 GB RAM,
and limits numerical libraries to one thread per analysis worker. The input
directory must include all six CSVs and the from-training generation manifest
with its evidence; earlier checkpoint-based inputs fail the provenance gate.

The generation `COMPLETE` marker requires two matching full fitting and
sampling sequences. The analysis `COMPLETE` marker additionally requires all
numerical checks and a successfully rendered report. The reference run completed
both full fitting sequences, all eight comparisons and 24 estimator fits, the
null diagnostic, seven figures, and the validated ten-page report. Its
[runtime evidence](../data/evidence/runtime_environment.json) records the
actual GPU hardware, driver, libraries, and preflight checks.
