#!/usr/bin/env bash
#SBATCH --job-name=ibdmdb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm-%x-%A_%a.log
set -euo pipefail

# Submit from the repository root. Add your site's module loads here, or export
# absolute Python/R paths before submission. Edit partition/account/time as needed.
cd "${SLURM_SUBMIT_DIR:?Submit this job from the TwoSample repository root}"
stage="${1:?Expected prepare, generator, or analysis}"
if [[ "$stage" == generator ]]; then
  models=(d dt icfm mbgan)
  stage="${models[${SLURM_ARRAY_TASK_ID:?Missing generator array index}]}"
fi
threads="${GENERATOR_THREADS:-2}"
export OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
args=(--stage "$stage"
      --output "${IBDMDB_OUTPUT:-$PWD/output/ibdmdb_study}"
      --r-lib "${IBDMDB_R_LIB:-$PWD/output/ibdmdb_study/r_lib}"
      --python-generators "${GENERATOR_PYTHON:-$PWD/.venv-generators/bin/python}"
      --python-mbgan "${MBGAN_PYTHON:-$PWD/.venv-mbgan/bin/python}"
      --rscript "${IBDMDB_RSCRIPT:-Rscript}"
      --threads "$threads" --workers "${SLURM_CPUS_PER_TASK:-8}"
      --icfm-steps "${ICFM_STEPS:-20000}" --mbgan-iterations "${MBGAN_ITERATIONS:-500000}"
      --n-synthetic "${N_SYNTHETIC:-1000}"
      --icfm-device "${ICFM_DEVICE:-cpu}" --mbgan-device "${MBGAN_DEVICE:-cpu}")
if [[ -n "${EXPERIMENT_HUB_CACHE:-}" ]]; then args+=(--cache-dir "$EXPERIMENT_HUB_CACHE"); fi
if [[ "${IBDMDB_OFFLINE:-0}" == 1 ]]; then args+=(--offline); fi
driver="${DRIVER_PYTHON:-python3}"
if [[ "$stage" == analysis ]]; then
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  for analysis_stage in batts figures report; do
    "$driver" scripts/run_ibdmdb_experiment.py "${args[@]:2}" --stage "$analysis_stage"
  done
else
  "$driver" scripts/run_ibdmdb_experiment.py "${args[@]}"
fi
