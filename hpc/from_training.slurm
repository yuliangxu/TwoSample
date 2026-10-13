#!/usr/bin/env bash
#SBATCH --job-name=twosample-historical-mbgan
#SBATCH --account=mastatlab
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${TWO_SAMPLE_SOURCE_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=1 TF_DETERMINISTIC_OPS=1
export TWO_SAMPLE_RUN_ROOT="${TWO_SAMPLE_RUN_ROOT:-$PWD/output/from_training_analysis}"
export TWO_SAMPLE_GENERATION_ROOT="${TWO_SAMPLE_GENERATION_ROOT:-$PWD/output/from_training_generators}"
export TWO_SAMPLE_EXECUTION_ROOT="${TWO_SAMPLE_EXECUTION_ROOT:-$PWD/output/from_training_execution}"
export TWO_SAMPLE_RUNTIME_PROVENANCE="${TWO_SAMPLE_RUNTIME_PROVENANCE:-$TWO_SAMPLE_EXECUTION_ROOT/runtime_environment.json}"
export MBGAN_PYTHON="${MBGAN_PYTHON:-$PWD/.venv-mbgan/bin/python}"
export MPLCONFIGDIR="$TWO_SAMPLE_EXECUTION_ROOT/matplotlib"
"$MBGAN_PYTHON" hpc/runtime_preflight.py
"${GENERATOR_PYTHON:-python3}" -m generators.reproduce \
  --tensorflow-python "$MBGAN_PYTHON" \
  --replica-workers 2 --output "$TWO_SAMPLE_GENERATION_ROOT"
# Successful paired training and source-bound evidence must exist before any
# downstream fits start. The analysis then applies its complete manifest gate.
test -f "$TWO_SAMPLE_GENERATION_ROOT/COMPLETE"
test -s "$TWO_SAMPLE_GENERATION_ROOT/data/generation_manifest.json"
export TWO_SAMPLE_DATA_DIR="$TWO_SAMPLE_GENERATION_ROOT/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${CASE_STUDY_RSCRIPT:-Rscript}" scripts/reproduce_case_study_batts.R "${SLURM_CPUS_PER_TASK:-1}"
test -f "$TWO_SAMPLE_RUN_ROOT/ANALYSIS_COMPLETE"
"${REPORT_PYTHON:-$PWD/.venv-report/bin/python}" scripts/build_report.py "$TWO_SAMPLE_RUN_ROOT"
test -f "$TWO_SAMPLE_RUN_ROOT/COMPLETE"
