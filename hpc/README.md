# Slurm execution

Prepare the external input directory as described in [PREPROCESSING.md](../PREPROCESSING.md).
Install the separate generator and TensorFlow environments from the root
[README](../README.md). Submit from the repository root, with an absolute input
directory and a new absolute output directory:

```sh
export GENERATOR_PYTHON=/absolute/environment-generators/bin/python
export MBGAN_PYTHON=/absolute/environment-mbgan/bin/python
export CASE_STUDY_RSCRIPT=/absolute/path/to/Rscript
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION \
  hpc/from_training.slurm /absolute/prepared-data /absolute/new-run
```

Adjust the account, partition, CPU allocation, memory, wall time, and R library
environment for your cluster. The template requests eight CPUs, 48 GB memory,
and one day; it does not configure an institution-specific software stack.
The full run trains two independent replicas of every generator, including
500,000 MBGAN updates per replica, before fitting BATTS and making the figures.
Check the job exit status and runtime completion markers; submission is not
completion. All data, model checkpoints, fit objects, and plots are written to
the external run directory.
