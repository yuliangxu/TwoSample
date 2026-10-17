#!/usr/bin/env bash
# Train every generator twice, verify replay, fit BATTS, and generate seven PDFs.
set -euo pipefail
if [[ $# -ne 2 || ${1:-} == --help ]]; then
  echo "Usage: MBGAN_PYTHON=/path/to/python scripts/run_pipeline.sh /absolute/prepared-data /absolute/new-run"
  echo "Optional: GENERATOR_PYTHON, CASE_STUDY_RSCRIPT, WORKERS (default 1)."
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi
case "$1" in /*) ;; *) echo "Prepared data path must be absolute." >&2; exit 2;; esac
case "$2" in /*) ;; *) echo "New run path must be absolute." >&2; exit 2;; esac
input_dir=$1
run_dir=$2
if [[ ! -d "$input_dir" || -e "$run_dir" ]]; then
  echo "Prepared data must exist and the run destination must not exist." >&2
  exit 2
fi
: "${MBGAN_PYTHON:?Set MBGAN_PYTHON to the Python executable in the separate TensorFlow environment}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export TF_ENABLE_ONEDNN_OPTS=1 TF_DETERMINISTIC_OPS=1
export TWO_SAMPLE_DATA_DIR="$input_dir"
export TWO_SAMPLE_PYTHON="${GENERATOR_PYTHON:-python3}"
"$TWO_SAMPLE_PYTHON" - "$input_dir" "$run_dir" <<'PY'
import sys
from generators.cohort import external_path
external_path(sys.argv[1], "Prepared data")
external_path(sys.argv[2], "Run directory")
PY
mkdir -p "$run_dir"
export MPLCONFIGDIR="$run_dir/matplotlib"
"$TWO_SAMPLE_PYTHON" -m generators.reproduce --data-dir "$input_dir" \
  --tensorflow-python "$MBGAN_PYTHON" --replica-workers 2 \
  --output "$run_dir/generators"
test -f "$run_dir/generators/COMPLETE"
export TWO_SAMPLE_DATA_DIR="$run_dir/generators/data"
export TWO_SAMPLE_RUN_ROOT="$run_dir/analysis"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${CASE_STUDY_RSCRIPT:-Rscript}" scripts/reproduce_case_study_batts.R "${WORKERS:-1}"
test -f "$TWO_SAMPLE_RUN_ROOT/ANALYSIS_COMPLETE"
echo "Completed analysis and figures: $TWO_SAMPLE_RUN_ROOT"
