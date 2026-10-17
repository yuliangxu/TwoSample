#!/usr/bin/env bash
#SBATCH --job-name=twosample-code
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export WORKERS="${SLURM_CPUS_PER_TASK:-1}"
bash scripts/run_pipeline.sh "$@"
