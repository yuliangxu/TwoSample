#!/usr/bin/env bash
#SBATCH --job-name=twosample-final
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
# Add the account, partition and module loads required by your cluster.
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TWO_SAMPLE_RUN_ROOT="${TWO_SAMPLE_RUN_ROOT:-$PWD/output/final}"
Rscript scripts/reproduce_case_study_batts.R "${SLURM_CPUS_PER_TASK:-1}"
"${REPORT_PYTHON:-$PWD/.venv-report/bin/python}" scripts/build_report.py "$TWO_SAMPLE_RUN_ROOT"
